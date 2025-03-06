"""
Base Agent Class for Legion AGI System

This module defines the foundational Agent class used throughout the system.
Agents are autonomous entities capable of reasoning, memory, and collaborative problem-solving.
"""

import uuid
import numpy as np
import json
import datetime
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
from loguru import logger

import ollama

from legion_agi.core.memory_module import MemoryModule
from legion_agi.core.global_workspace import GlobalWorkspace, InformationUnit
from legion_agi.utils.db_manager import DatabaseManager
from legion_agi.config import (
    DEFAULT_MODEL,
    INFERENCE_TEMPERATURE,
    INFERENCE_TOP_P,
    MAX_TOKENS
)


class Agent:
    """
    Base agent class for Legion AGI system.
    Provides core functionality for autonomous reasoning and collaboration.
    """
    
    def __init__(
        self,
        name: str,
        role: str,
        backstory: str,
        style: str,
        instructions: str,
        model: str = DEFAULT_MODEL,
        db_manager: Optional[DatabaseManager] = None,
        memory_module: Optional[MemoryModule] = None,
        global_workspace: Optional[GlobalWorkspace] = None
    ):
        """
        Initialize agent.
        
        Args:
            name: Agent name
            role: Agent's role/specialty
            backstory: Agent's backstory/context
            style: Communication style
            instructions: Specific instructions for this agent
            model: LLM model to use
            db_manager: Database manager instance
            memory_module: Memory module instance
            global_workspace: Global workspace instance
        """
        self.agent_id = str(uuid.uuid4())
        self.name = name
        self.role = role
        self.backstory = backstory
        self.style = style
        self.instructions = instructions
        self.model = model
        self.agent_list: List['Agent'] = []
        self.prompt: str = ""
        
        # Connect to services
        self.db_manager = db_manager
        self.memory_module = memory_module or MemoryModule(name=f"{name}_memory")
        self.global_workspace = global_workspace
        
        # Internal state
        self.reasoning_state: Dict[str, Any] = {
            "current_goal": None,
            "reasoning_chain": [],
            "iteration": 0,
            "reasoning_depth": 0,
            "working_memory": {}
        }
        
        # Cognitive parameters
        self.creativity = np.random.uniform(0.3, 0.9)  # Controls exploration vs. exploitation
        self.attention_span = np.random.uniform(0.5, 1.0)  # Controls focus
        self.learning_rate = np.random.uniform(0.1, 0.5)  # Controls adaptation speed
        
        # Register with database if available
        if self.db_manager:
            self.db_manager.save_agent(self)
            
        logger.info(f"Agent '{name}' ({role}) initialized with ID {self.agent_id}")
        
    def __repr__(self) -> str:
        return f"Agent(name={self.name}, role={self.role})"
        
    def set_agent_list(self, agent_list: List['Agent']) -> None:
        """
        Set the list of agents this agent can collaborate with.
        
        Args:
            agent_list: List of agent instances
        """
        self.agent_list = agent_list
        self.prompt = self.create_prompt()
        
    def create_prompt(self) -> str:
        """
        Create the agent's base prompt.
        
        Returns:
            Formatted prompt string
        """
        agent_names = ', '.join(
            [agent.name for agent in self.agent_list if agent.name != self.name]
        )
        
        persona = (
            f"You are {self.name}, {self.role}.\n"
            f"Backstory: {self.backstory}\n"
            f"Communication Style: {self.style}\n"
            f"{self.instructions}\n"
            f"Participants in the conversation: {agent_names}.\n"
            "Your task is to collaboratively brainstorm and build upon the previous "
            "discussions to contribute to a comprehensive solution.\n"
            "Respond in first person singular.\n"
            "Do not mention that you are an AI language model.\n"
            "Stay focused on the original question and avoid going off-topic.\n"
            "Think step by step and explain your reasoning process.\n"
        )
        return persona
        
    def process_global_broadcast(self, information_units: List[InformationUnit]) -> None:
        """
        Process information broadcast from the global workspace.
        
        Args:
            information_units: List of information units from broadcast
        """
        if not self.global_workspace:
            return
            
        # Store in agent's memory
        for info in information_units:
            # Convert to appropriate format for memory module
            memory_content = info.content
            memory_metadata = {
                "source": info.source,
                "broadcast_time": datetime.datetime.now().isoformat(),
                "broadcast_count": info.broadcast_count,
                **info.metadata
            }
            
            # Store in memory with importance proportional to activation
            self.memory_module.add_to_stm(
                content=memory_content,
                source=info.source,
                metadata=memory_metadata,
                importance=info.activation
            )
            
            # Also update working memory in reasoning state
            if isinstance(memory_content, dict):
                self.reasoning_state["working_memory"].update(memory_content)
            elif isinstance(memory_content, str):
                # For string content, use a simple hash as key
                key = f"broadcast_{hash(memory_content) % 10000}"
                self.reasoning_state["working_memory"][key] = memory_content
                
        logger.debug(f"Agent {self.name} processed {len(information_units)} broadcast units")
        
    def broadcast_to_global_workspace(self, 
                                    content: Any, 
                                    activation: float = 0.8,
                                    metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Broadcast information to the global workspace.
        
        Args:
            content: Information content to broadcast
            activation: Activation level (0.0 to 1.0)
            metadata: Additional metadata about the information
        """
        if not self.global_workspace:
            return
            
        # Inject information into global workspace
        self.global_workspace.inject_information(
            content=content,
            source=self.name,
            activation=activation,
            metadata=metadata or {}
        )
        
        logger.debug(f"Agent {self.name} broadcast information to global workspace")
        
    def respond(self,
              conversation_history: List[Dict[str, str]],
              temperature: float = INFERENCE_TEMPERATURE,
              top_p: float = INFERENCE_TOP_P,
              max_tokens: int = MAX_TOKENS) -> str:
        """
        Generate a response based on the conversation history.
        
        Args:
            conversation_history: History of the conversation
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        # Format conversation history
        formatted_history = self.format_conversation_history(conversation_history)
        
        # Retrieve relevant memories
        relevant_memories = self._retrieve_relevant_memories(formatted_history)
        memory_context = self._format_memories_for_context(relevant_memories)
        
        # Build full prompt
        full_prompt = (
            f"{self.prompt}\n\n"
            f"Your relevant memories and knowledge:\n{memory_context}\n\n"
            f"Conversation history:\n{formatted_history}\n\n"
            f"{self.name}:"
        )
        
        # Generate response
        response_content = self.generate_response(full_prompt, temperature, top_p, max_tokens)
        
        # Save to conversation history if DB manager is available
        if self.db_manager:
            self.db_manager.save_conversation(self.agent_id, self.name, response_content)
            
        # Add to memory
        self.memory_module.add_to_stm(
            content=response_content,
            source="self",
            metadata={
                "type": "response",
                "timestamp": datetime.datetime.now().isoformat()
            },
            importance=0.7  # Higher importance for own responses
        )
        
        # Broadcast to global workspace if available
        if self.global_workspace:
            self.broadcast_to_global_workspace(
                content=response_content,
                activation=0.8,
                metadata={
                    "type": "agent_response",
                    "agent_role": self.role
                }
            )
            
        return response_content
        
    def format_conversation_history(self,
                                  conversation_history: List[Dict[str, str]],
                                  limit: Optional[int] = None) -> str:
        """
        Format conversation history for inclusion in prompts.
        
        Args:
            conversation_history: List of conversation entries
            limit: Maximum number of entries to include (from the end)
            
        Returns:
            Formatted conversation history string
        """
        history_entries = conversation_history[-limit:] if limit else conversation_history
        
        formatted_history = ""
        for entry in history_entries:
            role = entry.get('role', '')
            content = entry.get('content', '')
            formatted_history += f"{role}: {content}\n\n"
            
        return formatted_history.strip()
        
    def _retrieve_relevant_memories(self, context: str) -> List[Dict[str, Any]]:
        """
        Retrieve memories relevant to the current context.
        
        Args:
            context: Current conversation context
            
        Returns:
            List of relevant memory items
        """
        # Query memory module for relevant memories
        relevant_memories = self.memory_module.recall_memory(
            query=context,
            memory_type="all",
            max_results=5
        )
        
        # Get associated memories for the top memory
        if relevant_memories and len(relevant_memories) > 0:
            top_memory = relevant_memories[0]
            associated = self.memory_module.get_associated_memories(top_memory['id'])
            
            # Add associated memories that aren't already included
            for memory in associated:
                if memory['id'] not in [m['id'] for m in relevant_memories]:
                    relevant_memories.append(memory)
                    
            # Limit to top 7 (Miller's Law)
            relevant_memories = relevant_memories[:7]
            
        return relevant_memories
        
    def _format_memories_for_context(self, memories: List[Dict[str, Any]]) -> str:
        """
        Format retrieved memories for inclusion in context.
        
        Args:
            memories: List of memory items
            
        Returns:
            Formatted memory context string
        """
        if not memories:
            return "No relevant memories."
            
        memory_texts = []
        for memory in memories:
            content = memory['content']
            source = memory['source']
            timestamp = memory.get('timestamp', 'unknown time')
            
            # Format the memory
            memory_text = f"- Memory from {source} at {timestamp}: {content}"
            memory_texts.append(memory_text)
            
        return "\n".join(memory_texts)
        
    def generate_response(self, 
                         prompt: str,
                         temperature: float = INFERENCE_TEMPERATURE,
                         top_p: float = INFERENCE_TOP_P,
                         max_tokens: int = MAX_TOKENS) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        messages = [{'role': 'user', 'content': prompt}]
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    'temperature': temperature,
                    'top_p': top_p,
                    'num_predict': max_tokens
                }
            )
            
            response_content = response['message']['content'].strip()
            
            # Add to memory as experience
            self.memory_module.replay_buffer.append({
                'content': response_content,
                'prompt': prompt,
                'timestamp': datetime.datetime.now().isoformat(),
                'model': self.model,
                'temperature': temperature,
                'importance': 0.6  # Moderate importance
            })
            
            return response_content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while processing your request. {str(e)}"
            
    def step_reasoning(self, 
                      goal: str, 
                      context: str,
                      reasoning_method: str = "chain_of_thought") -> Dict[str, Any]:
        """
        Perform a single step of reasoning toward a goal.
        
        Args:
            goal: Reasoning goal/question
            context: Context for reasoning
            reasoning_method: Method of reasoning to use
            
        Returns:
            Result of the reasoning step
        """
        # Update reasoning state
        if self.reasoning_state["current_goal"] != goal:
            # New goal, reset reasoning state
            self.reasoning_state["current_goal"] = goal
            self.reasoning_state["reasoning_chain"] = []
            self.reasoning_state["iteration"] = 0
            self.reasoning_state["reasoning_depth"] = 0
            
        # Increment iteration
        self.reasoning_state["iteration"] += 1
        
        # Build reasoning prompt based on method
        if reasoning_method == "chain_of_thought":
            reasoning_prompt = self._build_cot_prompt(goal, context)
        elif reasoning_method == "tree_of_thought":
            reasoning_prompt = self._build_tot_prompt(goal, context)
        else:
            reasoning_prompt = self._build_default_prompt(goal, context)
            
        # Generate reasoning step
        reasoning_output = self.generate_response(
            reasoning_prompt,
            temperature=0.7,  # Slightly higher temperature for creative reasoning
            top_p=0.9,
            max_tokens=1000
        )
        
        # Process reasoning output
        result = {
            "goal": goal,
            "iteration": self.reasoning_state["iteration"],
            "reasoning_output": reasoning_output,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add to reasoning chain
        self.reasoning_state["reasoning_chain"].append(result)
        
        # Add to memory with high importance
        self.memory_module.add_to_stm(
            content=reasoning_output,
            source="reasoning",
            metadata={
                "type": "reasoning_step",
                "goal": goal,
                "iteration": self.reasoning_state["iteration"],
                "method": reasoning_method
            },
            importance=0.8  # High importance for reasoning
        )
        
        return result
        
    def _build_cot_prompt(self, goal: str, context: str) -> str:
        """
        Build a Chain-of-Thought reasoning prompt.
        
        Args:
            goal: Reasoning goal/question
            context: Context for reasoning
            
        Returns:
            Formatted reasoning prompt
        """
        # Get previous reasoning steps if available
        previous_steps = ""
        if self.reasoning_state["reasoning_chain"]:
            for i, step in enumerate(self.reasoning_state["reasoning_chain"]):
                previous_steps += f"Step {i+1}: {step['reasoning_output']}\n\n"
                
        # Build prompt
        prompt = (
            f"{self.prompt}\n\n"
            f"I want you to solve the following problem using Chain-of-Thought reasoning:\n"
            f"Problem: {goal}\n\n"
            f"Context:\n{context}\n\n"
        )
        
        if previous_steps:
            prompt += (
                f"Previous reasoning steps:\n{previous_steps}\n"
                f"Continue the reasoning process with the next step. Focus on making progress "
                f"toward solving the problem. Think carefully and show your work.\n\n"
                f"Step {self.reasoning_state['iteration'] + 1}:"
            )
        else:
            prompt += (
                f"Think through this step by step. Break down the problem, identify the key "
                f"components, and systematically work toward a solution. Show your work.\n\n"
                f"Step 1:"
            )
            
        return prompt
        
    def _build_tot_prompt(self, goal: str, context: str) -> str:
        """
        Build a Tree-of-Thought reasoning prompt.
        
        Args:
            goal: Reasoning goal/question
            context: Context for reasoning
            
        Returns:
            Formatted reasoning prompt
        """
        # Get previous reasoning steps
        previous_steps = ""
        if self.reasoning_state["reasoning_chain"]:
            for i, step in enumerate(self.reasoning_state["reasoning_chain"]):
                previous_steps += f"Step {i+1}: {step['reasoning_output']}\n\n"
                
        # Build prompt
        prompt = (
            f"{self.prompt}\n\n"
            f"I want you to solve the following problem using Tree-of-Thought reasoning:\n"
            f"Problem: {goal}\n\n"
            f"Context:\n{context}\n\n"
        )
        
        if previous_steps:
            prompt += (
                f"Previous reasoning steps:\n{previous_steps}\n"
                f"Continue the reasoning process with the next step. Consider multiple paths "
                f"forward and evaluate which one is most promising. Generate at least 2-3 "
                f"possible paths, then select the most promising one to explore further.\n\n"
                f"Step {self.reasoning_state['iteration'] + 1}:"
            )
        else:
            prompt += (
                f"Consider multiple approaches to this problem. Generate 2-3 possible starting "
                f"points, evaluate their potential, and select the most promising one to "
                f"explore further. Show your reasoning.\n\n"
                f"Step 1:"
            )
            
        return prompt
        
    def _build_default_prompt(self, goal: str, context: str) -> str:
        """
        Build a default reasoning prompt.
        
        Args:
            goal: Reasoning goal/question
            context: Context for reasoning
            
        Returns:
            Formatted reasoning prompt
        """
        # Build simple reasoning prompt
        prompt = (
            f"{self.prompt}\n\n"
            f"I want you to help me with the following problem:\n"
            f"{goal}\n\n"
            f"Context:\n{context}\n\n"
            f"Please provide your thoughts on this problem:"
        )
        return prompt
        
    def collaborate_with(self, other_agent: 'Agent', 
                        topic: str, 
                        context: str = "") -> Dict[str, str]:
        """
        Collaborate with another agent on a topic.
        
        Args:
            other_agent: Agent to collaborate with
            topic: Topic to collaborate on
            context: Additional context
            
        Returns:
            Dictionary with both agents' contributions
        """
        # Generate initial thoughts on the topic
        my_thoughts = self.step_reasoning(topic, context)["reasoning_output"]
        
        # Share with other agent
        collaboration_context = (
            f"Topic: {topic}\n"
            f"Context: {context}\n"
            f"{self.name}'s thoughts: {my_thoughts}"
        )
        
        # Get other agent's perspective
        other_perspective = other_agent.step_reasoning(
            topic, collaboration_context)["reasoning_output"]
            
        # Integrate other agent's perspective
        integration_context = (
            f"Topic: {topic}\n"
            f"Context: {context}\n"
            f"Your initial thoughts: {my_thoughts}\n"
            f"{other_agent.name}'s perspective: {other_perspective}\n"
            f"Please integrate {other_agent.name}'s perspective with your own thinking:"
        )
        
        integrated_response = self.generate_response(integration_context)
        
        # Create record of collaboration
        collaboration_record = {
            'topic': topic,
            'context': context,
            f'{self.name}_initial': my_thoughts,
            f'{other_agent.name}_perspective': other_perspective,
            'integrated_result': integrated_response
        }
        
        # Add collaboration to memory
        self.memory_module.add_to_stm(
            content=collaboration_record,
            source="collaboration",
            metadata={
                "type": "collaboration",
                "collaborator": other_agent.name,
                "topic": topic
            },
            importance=0.9  # High importance for collaborations
        )
        
        # Other agent also remembers
        other_agent.memory_module.add_to_stm(
            content=collaboration_record,
            source="collaboration",
            metadata={
                "type": "collaboration",
                "collaborator": self.name,
                "topic": topic
            },
            importance=0.9
        )
        
        return collaboration_record
        
    def update_cognitive_parameters(self, 
                                   creativity: Optional[float] = None,
                                   attention_span: Optional[float] = None,
                                   learning_rate: Optional[float] = None) -> None:
        """
        Update agent's cognitive parameters.
        
        Args:
            creativity: New creativity value (0.0 to 1.0)
            attention_span: New attention span value (0.0 to 1.0)
            learning_rate: New learning rate value (0.0 to 1.0)
        """
        if creativity is not None:
            self.creativity = max(0.0, min(1.0, creativity))
            
        if attention_span is not None:
            self.attention_span = max(0.0, min(1.0, attention_span))
            
        if learning_rate is not None:
            self.learning_rate = max(0.0, min(1.0, learning_rate))
            
        logger.debug(f"Updated cognitive parameters for {self.name}: "
                   f"creativity={self.creativity}, attention_span={self.attention_span}, "
                   f"learning_rate={self.learning_rate}")
                   
    def save_agent_state(self, filename: str) -> None:
        """
        Save agent state to file.
        
        Args:
            filename: Path to save state to
        """
        state = {
            'agent_id': self.agent_id,
            'name': self.name,
            'role': self.role,
            'backstory': self.backstory,
            'style': self.style,
            'instructions': self.instructions,
            'model': self.model,
            'cognitive_parameters': {
                'creativity': self.creativity,
                'attention_span': self.attention_span,
                'learning_rate': self.learning_rate
            },
            'reasoning_state': self.reasoning_state,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"Agent state saved to {filename}")
        
        # Also save memory state
        memory_filename = filename.replace('.json', '_memory.pkl')
        self.memory_module.save_memory_state(memory_filename)
        
    def load_agent_state(self, filename: str) -> None:
        """
        Load agent state from file.
        
        Args:
            filename: Path to load state from
        """
        with open(filename, 'r') as f:
            state = json.load(f)
            
        # Update agent properties
        self.agent_id = state['agent_id']
        self.name = state['name']
        self.role = state['role']
        self.backstory = state['backstory']
        self.style = state['style']
        self.instructions = state['instructions']
        self.model = state['model']
        
        # Update cognitive parameters
        cp = state.get('cognitive_parameters', {})
        self.creativity = cp.get('creativity', self.creativity)
        self.attention_span = cp.get('attention_span', self.attention_span)
        self.learning_rate = cp.get('learning_rate', self.learning_rate)
        
        # Update reasoning state
        self.reasoning_state = state.get('reasoning_state', self.reasoning_state)
        
        logger.info(f"Agent state loaded from {filename}")
        
        # Also load memory state if available
        memory_filename = filename.replace('.json', '_memory.pkl')
        try:
            self.memory_module.load_memory_state(memory_filename)
        except:
            logger.warning(f"Could not load memory state from {memory_filename}")

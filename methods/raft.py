"""
RAFT (Reasoning, Analysis, Feedback, and Thought) Method

This module implements the RAFT reasoning method which facilitates deeper reasoning
through iterative critique and refinement of ideas. It enables agents to exchange
feedback and build upon each other's thoughts.
"""

import ollama
import random
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from legion_agi.agents.agent_base import Agent
from legion_agi.core.global_workspace import GlobalWorkspace, InformationUnit
from legion_agi.utils.db_manager import DatabaseManager
from legion_agi.config import DEFAULT_MODEL, RAFT_ITERATIONS


class RAFTMethod:
    """
    Implementation of the RAFT reasoning method.
    
    RAFT stands for:
    - Reasoning: Initial reasoning process for the problem
    - Analysis: Analyzing and challenging the reasoning
    - Feedback: Providing structured feedback on approaches
    - Thought: Deeper integration and reflection on the reasoning process
    """
    
    def __init__(
        self,
        agents: List[Agent],
        global_workspace: Optional[GlobalWorkspace] = None,
        db_manager: Optional[DatabaseManager] = None,
        model: str = DEFAULT_MODEL
    ):
        """
        Initialize RAFT method.
        
        Args:
            agents: List of participating agents
            global_workspace: Optional global workspace for coordination
            db_manager: Optional database manager for persistence
            model: LLM model for fallback reasoning
        """
        self.agents = agents
        self.global_workspace = global_workspace
        self.db_manager = db_manager
        self.model = model
        
        # Internal state
        self.question: Optional[str] = None
        self.stage = "init"  # init, reasoning, analysis, feedback, thought
        self.reasoning_chains: Dict[str, List[Dict[str, Any]]] = {}
        self.analyses: List[Dict[str, Any]] = []
        self.feedback: List[Dict[str, Any]] = []
        self.integrated_thoughts: List[Dict[str, Any]] = []
        self.final_synthesis: Optional[Dict[str, Any]] = None
        
        logger.info(f"RAFT method initialized with {len(agents)} agents")
        
    def set_question(self, question: str) -> None:
        """
        Set the question for the RAFT process.
        
        Args:
            question: The question or problem statement
        """
        self.question = question
        self.stage = "reasoning"
        
        # Reset state
        self.reasoning_chains = {}
        self.analyses = []
        self.feedback = []
        self.integrated_thoughts = []
        self.final_synthesis = None
        
        logger.info(f"RAFT method question set: {question[:100]}...")
        
    def generate_reasoning_chains(self, 
                                iterations: int = RAFT_ITERATIONS) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate initial reasoning chains from each agent.
        
        Args:
            iterations: Number of reasoning iterations
            
        Returns:
            Dictionary mapping agent IDs to their reasoning chains
        """
        if not self.question:
            raise ValueError("Question not set")
            
        logger.info(f"Generating reasoning chains with {len(self.agents)} agents")
        
        # Move to reasoning stage
        self.stage = "reasoning"
        
        # Generate reasoning chains from each agent
        for agent in self.agents:
            # Skip system agents for direct reasoning
            if agent.name in ["GuidingAgent", "RefinementAgent", "EvaluationAgent"]:
                continue
                
            # Initialize reasoning chain for this agent
            self.reasoning_chains[agent.agent_id] = []
            
            # Generate reasoning steps
            for i in range(iterations):
                # Create reasoning prompt
                prev_reasoning = ""
                if i > 0 and self.reasoning_chains[agent.agent_id]:
                    # Include previous reasoning steps
                    for step in self.reasoning_chains[agent.agent_id]:
                        prev_reasoning += f"Step {step['iteration']}: {step['reasoning']}\n\n"
                        
                prompt = (
                    f"I want you to reason through the following question step by step:\n\n"
                    f"Question: {self.question}\n\n"
                    f"Your role: {agent.role}\n"
                    f"Your expertise: {agent.backstory}\n\n"
                )
                
                if prev_reasoning:
                    prompt += (
                        f"Your previous reasoning steps:\n{prev_reasoning}\n\n"
                        f"Continue your reasoning process with step {i+1}. Focus on building upon "
                        f"your previous steps and exploring new aspects of the problem. Be specific, "
                        f"rigorous, and show your thinking process clearly."
                    )
                else:
                    prompt += (
                        f"Start your reasoning process with step 1. Break down the problem, identify "
                        f"key assumptions, and begin developing your approach. Be specific, rigorous, "
                        f"and show your thinking process clearly."
                    )
                    
                # Generate reasoning
                reasoning = agent.generate_response(
                    prompt, temperature=0.5)
                    
                # Store reasoning step
                reasoning_step = {
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "agent_role": agent.role,
                    "iteration": i + 1,
                    "reasoning": reasoning,
                    "timestamp": str(agent.memory_module.short_term_memory[-1]['timestamp'] 
                                  if agent.memory_module.short_term_memory else "unknown")
                }
                
                self.reasoning_chains[agent.agent_id].append(reasoning_step)
                
                # Broadcast to global workspace if available
                if self.global_workspace:
                    self.global_workspace.inject_information(
                        content={
                            "agent_name": agent.name,
                            "agent_role": agent.role,
                            "iteration": i + 1,
                            "reasoning": reasoning
                        },
                        source=agent.name,
                        activation=0.7,
                        metadata={
                            "stage": "reasoning",
                            "method": "RAFT",
                            "agent_id": agent.agent_id,
                            "iteration": i + 1
                        }
                    )
                    
            logger.info(f"Generated reasoning chain for {agent.name} with {iterations} steps")
            
        return self.reasoning_chains
        
    def analyze_reasoning(self) -> List[Dict[str, Any]]:
        """
        Analyze and challenge the reasoning chains.
        
        Returns:
            List of analysis results
        """
        if not self.question:
            raise ValueError("Question not set")
            
        if not self.reasoning_chains:
            self.generate_reasoning_chains()
            
        logger.info("Analyzing reasoning chains")
        
        # Move to analysis stage
        self.stage = "analysis"
        
        # Each agent analyzes another agent's reasoning
        for analyzer in self.agents:
            # Skip system agents for direct analysis
            if analyzer.name in ["GuidingAgent", "RefinementAgent", "EvaluationAgent"]:
                continue
                
            # Select a target agent to analyze (not self)
            eligible_targets = [
                agent_id for agent_id in self.reasoning_chains.keys() 
                if agent_id != analyzer.agent_id
            ]
            
            if not eligible_targets:
                continue
                
            target_id = random.choice(eligible_targets)
            target_chain = self.reasoning_chains[target_id]
            
            if not target_chain:
                continue
                
            # Find the target agent name and role
            target_name = target_chain[0]["agent_name"]
            target_role = target_chain[0]["agent_role"]
            
            # Format target reasoning chain
            target_reasoning = ""
            for step in target_chain:
                target_reasoning += f"Step {step['iteration']}: {step['reasoning']}\n\n"
                
            # Create analysis prompt
            prompt = (
                f"Analyze and critically evaluate the following reasoning chain about this question:\n\n"
                f"Question: {self.question}\n\n"
                f"Reasoning chain by {target_name} ({target_role}):\n{target_reasoning}\n\n"
                f"As {analyzer.role}, analyze this reasoning chain for:\n"
                f"1. Strengths and insights\n"
                f"2. Weaknesses, gaps, or logical flaws\n"
                f"3. Hidden assumptions\n"
                f"4. Alternative perspectives that weren't considered\n"
                f"5. How the reasoning could be improved\n\n"
                f"Provide a thorough analysis that identifies both strengths and opportunities for improvement."
            )
            
            # Generate analysis
            analysis = analyzer.generate_response(prompt, temperature=0.4)
            
            # Store analysis
            analysis_entry = {
                "analyzer_id": analyzer.agent_id,
                "analyzer_name": analyzer.name,
                "analyzer_role": analyzer.role,
                "target_id": target_id,
                "target_name": target_name,
                "target_role": target_role,
                "analysis": analysis,
                "timestamp": str(analyzer.memory_module.short_term_memory[-1]['timestamp'] 
                              if analyzer.memory_module.short_term_memory else "unknown")
            }
            
            self.analyses.append(analysis_entry)
            
            # Broadcast to global workspace if available
            if self.global_workspace:
                self.global_workspace.inject_information(
                    content={
                        "analyzer_name": analyzer.name,
                        "analyzer_role": analyzer.role,
                        "target_name": target_name,
                        "target_role": target_role,
                        "analysis": analysis
                    },
                    source=analyzer.name,
                    activation=0.8,
                    metadata={
                        "stage": "analysis",
                        "method": "RAFT",
                        "analyzer_id": analyzer.agent_id,
                        "target_id": target_id
                    }
                )
                
        return self.analyses
        
    def provide_feedback(self) -> List[Dict[str, Any]]:
        """
        Provide structured feedback on the analyses.
        
        Returns:
            List of feedback results
        """
        if not self.question:
            raise ValueError("Question not set")
            
        if not self.analyses:
            self.analyze_reasoning()
            
        logger.info("Providing feedback on analyses")
        
        # Move to feedback stage
        self.stage = "feedback"
        
        # Each analyzed agent responds to their analysis
        for analysis in self.analyses:
            target_id = analysis["target_id"]
            target_name = analysis["target_name"]
            
            # Find the target agent
            target_agent = None
            for agent in self.agents:
                if agent.agent_id == target_id:
                    target_agent = agent
                    break
                    
            if not target_agent:
                continue
                
            # Format the analysis
            analyzer_name = analysis["analyzer_name"]
            analyzer_role = analysis["analyzer_role"]
            analysis_text = analysis["analysis"]
            
            # Create feedback prompt
            prompt = (
                f"You received the following analysis of your reasoning on this question:\n\n"
                f"Question: {self.question}\n\n"
                f"Analysis by {analyzer_name} ({analyzer_role}):\n{analysis_text}\n\n"
                f"As {target_agent.role}, reflect on this analysis and provide your feedback:\n"
                f"1. Which points in the analysis do you agree with?\n"
                f"2. Which points do you disagree with and why?\n"
                f"3. How would you refine your reasoning based on this feedback?\n"
                f"4. What new insights or approaches emerged from considering this analysis?\n\n"
                f"Provide thoughtful, constructive feedback that advances the collective understanding."
            )
            
            # Generate feedback
            feedback = target_agent.generate_response(prompt, temperature=0.5)
            
            # Store feedback
            feedback_entry = {
                "feedback_agent_id": target_agent.agent_id,
                "feedback_agent_name": target_agent.name,
                "feedback_agent_role": target_agent.role,
                "analyzer_id": analysis["analyzer_id"],
                "analyzer_name": analyzer_name,
                "analyzer_role": analyzer_role,
                "feedback": feedback,
                "timestamp": str(target_agent.memory_module.short_term_memory[-1]['timestamp'] 
                              if target_agent.memory_module.short_term_memory else "unknown")
            }
            
            self.feedback.append(feedback_entry)
            
            # Broadcast to global workspace if available
            if self.global_workspace:
                self.global_workspace.inject_information(
                    content={
                        "feedback_agent_name": target_agent.name,
                        "feedback_agent_role": target_agent.role,
                        "analyzer_name": analyzer_name,
                        "analyzer_role": analyzer_role,
                        "feedback": feedback
                    },
                    source=target_agent.name,
                    activation=0.8,
                    metadata={
                        "stage": "feedback",
                        "method": "RAFT",
                        "feedback_agent_id": target_agent.agent_id,
                        "analyzer_id": analysis["analyzer_id"]
                    }
                )
                
        return self.feedback
        
    def integrate_thought(self) -> List[Dict[str, Any]]:
        """
        Integrate feedback and generate deeper thoughts.
        
        Returns:
            List of integrated thought results
        """
        if not self.question:
            raise ValueError("Question not set")
            
        if not self.feedback:
            self.provide_feedback()
            
        logger.info("Integrating thoughts from feedback")
        
        # Move to thought stage
        self.stage = "thought"
        
        # Each agent that provided feedback now integrates it into deeper thought
        for feedback_entry in self.feedback:
            agent_id = feedback_entry["feedback_agent_id"]
            
            # Find the agent
            agent = None
            for a in self.agents:
                if a.agent_id == agent_id:
                    agent = a
                    break
                    
            if not agent:
                continue
                
            # Get the agent's reasoning chain
            reasoning_chain = self.reasoning_chains.get(agent_id, [])
            if not reasoning_chain:
                continue
                
            # Format the reasoning chain
            reasoning_text = ""
            for step in reasoning_chain:
                reasoning_text += f"Step {step['iteration']}: {step['reasoning']}\n\n"
                
            # Format the feedback
            analyzer_name = feedback_entry["analyzer_name"]
            analyzer_role = feedback_entry["analyzer_role"]
            feedback_text = feedback_entry["feedback"]
            
            # Create integration prompt
            prompt = (
                f"Integrate your reasoning and feedback on this question:\n\n"
                f"Question: {self.question}\n\n"
                f"Your original reasoning:\n{reasoning_text}\n\n"
                f"Your feedback to {analyzer_name}'s analysis:\n{feedback_text}\n\n"
                f"Now, synthesize a deeper, more refined understanding of the problem. Integrate the "
                f"original reasoning with the insights from the feedback exchange. Develop a more "
                f"nuanced, comprehensive perspective that addresses the strengths and weaknesses "
                f"identified. Focus on generating novel insights that emerge from this synthetic process."
            )
            
            # Generate integrated thought
            integrated_thought = agent.generate_response(prompt, temperature=0.6)
            
            # Store integrated thought
            thought_entry = {
                "agent_id": agent.agent_id,
                "agent_name": agent.name,
                "agent_role": agent.role,
                "analyzer_name": analyzer_name,
                "analyzer_role": analyzer_role,
                "integrated_thought": integrated_thought,
                "timestamp": str(agent.memory_module.short_term_memory[-1]['timestamp'] 
                              if agent.memory_module.short_term_memory else "unknown")
            }
            
            self.integrated_thoughts.append(thought_entry)
            
            # Broadcast to global workspace if available
            if self.global_workspace:
                self.global_workspace.inject_information(
                    content={
                        "agent_name": agent.name,
                        "agent_role": agent.role,
                        "integrated_thought": integrated_thought
                    },
                    source=agent.name,
                    activation=0.9,
                    metadata={
                        "stage": "thought",
                        "method": "RAFT",
                        "agent_id": agent.agent_id
                    }
                )
                
        return self.integrated_thoughts
        
    def synthesize_final_thought(self) -> Dict[str, Any]:
        """
        Synthesize all integrated thoughts into a final coherent solution.
        
        Returns:
            Final synthesis of thoughts
        """
        if not self.question:
            raise ValueError("Question not set")
            
        if not self.integrated_thoughts:
            self.integrate_thought()
            
        logger.info("Synthesizing final thought")
        
        # Compile all integrated thoughts
        all_thoughts = ""
        for thought in self.integrated_thoughts:
            all_thoughts += f"{thought['agent_name']} ({thought['agent_role']}): {thought['integrated_thought']}\n\n"
            
        # Create synthesis prompt
        prompt = (
            f"Synthesize the following integrated thoughts into a comprehensive, coherent solution "
            f"to the original question:\n\n"
            f"Question: {self.question}\n\n"
            f"Integrated thoughts from experts:\n{all_thoughts}\n\n"
            f"Create a unified, sophisticated answer that represents the collective intelligence "
            f"of these experts. Incorporate the diverse perspectives while addressing tensions or "
            f"contradictions. Aim for a solution that is more powerful than any individual contribution, "
            f"highlighting emergent insights that arise from the integration process."
        )
        
        # Select the most appropriate agent for synthesis
        synthesis_agent = None
        
        # First try to find a suitable agent from the RAFT participants
        for agent in self.agents:
            # System agents may be more suitable for synthesis
            if agent.name in ["RefinementAgent", "EvaluationAgent"]:
                synthesis_agent = agent
                break
                
        # If no system agent, use the first available agent
        if not synthesis_agent and self.agents:
            synthesis_agent = self.agents[0]
            
        # Generate synthesis
        if synthesis_agent:
            final_synthesis = synthesis_agent.generate_response(prompt, temperature=0.4)
            synthesizer = f"{synthesis_agent.name} ({synthesis_agent.role})"
        else:
            # Fallback to direct LLM call
            messages = [{'role': 'user', 'content': prompt}]
            
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=messages,
                    options={'temperature': 0.4}
                )
                
                final_synthesis = response['message']['content'].strip()
                synthesizer = "System Synthesizer"
                
            except Exception as e:
                logger.error(f"Error synthesizing thoughts: {e}")
                final_synthesis = "Error synthesizing thoughts: " + str(e)
                synthesizer = "Error"
                
        # Store final synthesis
        self.final_synthesis = {
            "question": self.question,
            "final_synthesis": final_synthesis,
            "synthesizer": synthesizer,
            "contributors": [
                {"name": thought['agent_name'], "role": thought['agent_role']}
                for thought in self.integrated_thoughts
            ],
            "thought_count": len(self.integrated_thoughts),
            "stage": "thought",
            "method": "RAFT"
        }
        
        # Broadcast to global workspace if available
        if self.global_workspace:
            self.global_workspace.inject_information(
                content={
                    "question": self.question,
                    "final_synthesis": final_synthesis,
                    "method": "RAFT"
                },
                source="RAFT_method",
                activation=1.0,  # Highest activation for final synthesis
                metadata={
                    "stage": "thought",
                    "method": "RAFT",
                    "final_synthesis": True
                }
            )
            
            # Run global workspace cycles
            self.global_workspace.run_cycles(3)
            
        return self.final_synthesis
        
    def execute_full_method(self, 
                          question: str, 
                          iterations: int = RAFT_ITERATIONS) -> Dict[str, Any]:
        """
        Execute the full RAFT method on a question.
        
        Args:
            question: The question or problem
            iterations: Number of reasoning iterations
            
        Returns:
            Final synthesis of thoughts
        """
        logger.info(f"Executing full RAFT method on question: {question[:100]}...")
        
        # Set question
        self.set_question(question)
        
        # Stage 1: Reasoning - Generate reasoning chains
        reasoning_chains = self.generate_reasoning_chains(iterations=iterations)
        
        # Stage 2: Analysis - Analyze reasoning chains
        analyses = self.analyze_reasoning()
        
        # Stage 3: Feedback - Provide feedback on analyses
        feedback = self.provide_feedback()
        
        # Stage 4: Thought - Integrate thoughts
        thoughts = self.integrate_thought()
        
        # Final synthesis
        final_synthesis = self.synthesize_final_thought()
        
        return final_synthesis

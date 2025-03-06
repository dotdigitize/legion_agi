"""
PAST (Personas, Actions, Solutions, and Task) Method

This module implements the PAST reasoning method which organizes the problem-solving
process into four distinct phases, with specialized agents handling different
aspects of the solution.
"""

import ollama
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from legion_agi.agents.agent_base import Agent
from legion_agi.core.global_workspace import GlobalWorkspace, InformationUnit
from legion_agi.utils.db_manager import DatabaseManager
from legion_agi.config import DEFAULT_MODEL, PAST_DEPTH


class PASTMethod:
    """
    Implementation of the PAST reasoning method.
    
    PAST stands for:
    - Personas: Identify appropriate expert personas for the problem
    - Actions: Assign specific actions to each persona
    - Solutions: Develop solutions through collaborative reasoning
    - Task: Focus all efforts toward the original task/question
    """
    
    def __init__(
        self,
        agents: List[Agent],
        global_workspace: Optional[GlobalWorkspace] = None,
        db_manager: Optional[DatabaseManager] = None,
        model: str = DEFAULT_MODEL
    ):
        """
        Initialize PAST method.
        
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
        self.original_question: Optional[str] = None
        self.stage = "init"  # init, personas, actions, solutions, task
        self.assigned_actions: Dict[str, Dict[str, Any]] = {}
        self.solutions: List[Dict[str, Any]] = []
        self.final_solution: Optional[Dict[str, Any]] = None
        
        logger.info(f"PAST method initialized with {len(agents)} agents")
        
    def set_question(self, question: str) -> None:
        """
        Set the original question for the PAST process.
        
        Args:
            question: The original question or problem statement
        """
        self.original_question = question
        self.stage = "personas"
        
        # Reset state
        self.assigned_actions = {}
        self.solutions = []
        self.final_solution = None
        
        logger.info(f"PAST method question set: {question[:100]}...")
        
    def analyze_question(self) -> Dict[str, Any]:
        """
        Analyze the question to identify required expertises.
        
        Returns:
            Dictionary containing question analysis results
        """
        if not self.original_question:
            raise ValueError("Original question not set")
            
        logger.info("Analyzing question to identify required expertises")
        
        # Use LLM to analyze question
        prompt = (
            f"Analyze the following question to identify the expertise domains needed to answer it effectively:\n\n"
            f"Question: {self.original_question}\n\n"
            f"Please identify:\n"
            f"1. Primary domain of knowledge required\n"
            f"2. Secondary domains that would be helpful\n"
            f"3. Specific sub-areas of expertise that would be valuable\n"
            f"4. Types of reasoning needed (analytical, creative, systems-thinking, etc.)\n"
            f"5. Key concepts or terms that experts should understand\n\n"
            f"Format your response as a JSON object with these categories as keys."
        )
        
        messages = [{'role': 'user', 'content': prompt}]
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={'temperature': 0.3}  # Lower temperature for analytical task
            )
            
            analysis_text = response['message']['content'].strip()
            
            # Broadcast to global workspace if available
            if self.global_workspace:
                self.global_workspace.inject_information(
                    content={
                        "question_analysis": analysis_text,
                        "original_question": self.original_question
                    },
                    source="PAST_method",
                    activation=0.9,
                    metadata={
                        "stage": "personas",
                        "method": "PAST"
                    }
                )
                
            # Parse result
            try:
                # Try to extract JSON
                import json
                import re
                
                # Find JSON-like pattern
                json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
                if json_match:
                    analysis_json = json.loads(json_match.group(0))
                else:
                    # If we can't find JSON pattern, create simple structure
                    analysis_json = {
                        "analysis": analysis_text,
                        "primary_domain": "unknown",
                        "secondary_domains": [],
                        "expertise_areas": [],
                        "reasoning_types": [],
                        "key_concepts": []
                    }
            except:
                # Fallback if JSON parsing fails
                analysis_json = {
                    "analysis": analysis_text,
                    "primary_domain": "unknown",
                    "secondary_domains": [],
                    "expertise_areas": [],
                    "reasoning_types": [],
                    "key_concepts": []
                }
                
            return {
                "original_question": self.original_question,
                "analysis": analysis_json
            }
            
        except Exception as e:
            logger.error(f"Error analyzing question: {e}")
            return {
                "original_question": self.original_question,
                "error": str(e),
                "analysis": {
                    "primary_domain": "unknown",
                    "secondary_domains": [],
                    "expertise_areas": [],
                    "reasoning_types": [],
                    "key_concepts": []
                }
            }
            
    def assign_actions(self) -> Dict[str, Dict[str, Any]]:
        """
        Assign specific actions to each agent based on their expertise.
        
        Returns:
            Dictionary mapping agent IDs to assigned actions
        """
        if not self.original_question:
            raise ValueError("Original question not set")
            
        logger.info("Assigning actions to agents")
        
        # First get question analysis if we haven't already
        if self.stage == "personas":
            analysis = self.analyze_question()
        else:
            analysis = {"original_question": self.original_question}
            
        # Move to actions stage
        self.stage = "actions"
        
        # Generate actions for each agent
        for agent in self.agents:
            # Skip system agents
            if agent.name in ["GuidingAgent", "RefinementAgent", "EvaluationAgent"]:
                continue
                
            # Create action prompt for this agent
            prompt = (
                f"Based on the following question and your expertise as {agent.role}, "
                f"identify the specific actions you should take to help solve the problem.\n\n"
                f"Question: {self.original_question}\n\n"
                f"Your role: {agent.role}\n"
                f"Your expertise: {agent.backstory}\n\n"
                f"Please specify:\n"
                f"1. What specific aspects of the problem you'll focus on\n"
                f"2. What approach or methodology you'll use\n"
                f"3. What specific knowledge or techniques from your domain you'll apply\n"
                f"4. How your contribution will help solve the overall problem\n\n"
                f"Be specific and concrete about your planned actions."
            )
            
            # Generate action plan
            action_plan = agent.generate_response(prompt, temperature=0.4)
            
            # Store assigned action
            self.assigned_actions[agent.agent_id] = {
                "agent_id": agent.agent_id,
                "agent_name": agent.name,
                "agent_role": agent.role,
                "action_plan": action_plan,
                "timestamp": str(agent.memory_module.short_term_memory[-1]['timestamp'] 
                              if agent.memory_module.short_term_memory else "unknown")
            }
            
            # Broadcast to global workspace if available
            if self.global_workspace:
                self.global_workspace.inject_information(
                    content={
                        "agent_name": agent.name,
                        "agent_role": agent.role,
                        "action_plan": action_plan
                    },
                    source=agent.name,
                    activation=0.8,
                    metadata={
                        "stage": "actions",
                        "method": "PAST",
                        "agent_id": agent.agent_id
                    }
                )
                
        return self.assigned_actions
        
    def generate_solutions(self, depth: int = PAST_DEPTH) -> List[Dict[str, Any]]:
        """
        Generate solutions through collaborative agent reasoning.
        
        Args:
            depth: Depth of the solution generation process
            
        Returns:
            List of solution contributions from agents
        """
        if not self.original_question:
            raise ValueError("Original question not set")
            
        if not self.assigned_actions:
            self.assign_actions()
            
        logger.info(f"Generating solutions with {len(self.agents)} agents at depth {depth}")
        
        # Move to solutions stage
        self.stage = "solutions"
        
        # Generate solutions from each agent
        for i in range(depth):
            round_solutions = []
            
            for agent in self.agents:
                # Skip system agents for direct solution generation
                if agent.name in ["GuidingAgent", "RefinementAgent", "EvaluationAgent"]:
                    continue
                    
                # Get current context from previous solutions
                previous_solutions = ""
                for solution in self.solutions:
                    previous_solutions += f"{solution['agent_name']} ({solution['agent_role']}): {solution['solution']}\n\n"
                    
                # Create solution prompt for this agent
                action_plan = self.assigned_actions.get(agent.agent_id, {}).get('action_plan', 
                                                                            "No specific action plan assigned.")
                                                                            
                prompt = (
                    f"Based on the following question and your expertise, provide a solution contribution.\n\n"
                    f"Question: {self.original_question}\n\n"
                    f"Your role: {agent.role}\n"
                    f"Your action plan: {action_plan}\n\n"
                )
                
                if previous_solutions:
                    prompt += (
                        f"Previous solution contributions:\n{previous_solutions}\n\n"
                        f"Now, build upon these previous contributions and your own expertise to provide "
                        f"the next part of the solution. Focus on adding new insights or addressing "
                        f"aspects that haven't been covered yet. Be specific and direct in your contribution."
                    )
                else:
                    prompt += (
                        f"Provide your initial solution contribution based on your expertise. "
                        f"Be specific and direct in your approach, focusing on the aspects you're best "
                        f"qualified to address."
                    )
                    
                # Generate solution
                solution = agent.generate_response(prompt, temperature=0.5)
                
                # Store solution
                solution_entry = {
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "agent_role": agent.role,
                    "solution": solution,
                    "round": i + 1,
                    "timestamp": str(agent.memory_module.short_term_memory[-1]['timestamp'] 
                                  if agent.memory_module.short_term_memory else "unknown")
                }
                
                round_solutions.append(solution_entry)
                
                # Broadcast to global workspace if available
                if self.global_workspace:
                    self.global_workspace.inject_information(
                        content={
                            "agent_name": agent.name,
                            "agent_role": agent.role,
                            "solution": solution,
                            "round": i + 1
                        },
                        source=agent.name,
                        activation=0.8,
                        metadata={
                            "stage": "solutions",
                            "method": "PAST",
                            "agent_id": agent.agent_id,
                            "round": i + 1
                        }
                    )
                    
            # Add this round's solutions to the overall solutions
            self.solutions.extend(round_solutions)
            
            # Allow system agents to comment/guide if available
            for agent in self.agents:
                if agent.name in ["GuidingAgent", "RefinementAgent", "EvaluationAgent"]:
                    # Create prompt for system agent
                    all_solutions = ""
                    for solution in self.solutions:
                        all_solutions += f"{solution['agent_name']} ({solution['agent_role']}): {solution['solution']}\n\n"
                        
                    prompt = (
                        f"Review the following solution contributions for the question:\n\n"
                        f"Question: {self.original_question}\n\n"
                        f"Solutions so far:\n{all_solutions}\n\n"
                    )
                    
                    if agent.name == "GuidingAgent":
                        prompt += (
                            f"As the guiding agent, provide feedback to keep the discussion focused on the "
                            f"original question. Identify any aspects that are going off-track and suggest "
                            f"how to refocus the discussion."
                        )
                    elif agent.name == "RefinementAgent":
                        prompt += (
                            f"As the refinement agent, identify areas where the current solutions could be "
                            f"improved or expanded. Suggest specific refinements that would strengthen "
                            f"the overall solution."
                        )
                    elif agent.name == "EvaluationAgent":
                        prompt += (
                            f"As the evaluation agent, assess the solutions provided so far. "
                            f"Identify strengths, weaknesses, and gaps in the current approach. "
                            f"Provide constructive feedback for improvement."
                        )
                        
                    # Generate feedback
                    feedback = agent.generate_response(prompt, temperature=0.4)
                    
                    # Store feedback as a system solution
                    system_entry = {
                        "agent_id": agent.agent_id,
                        "agent_name": agent.name,
                        "agent_role": agent.role,
                        "solution": feedback,
                        "round": i + 1,
                        "system_feedback": True,
                        "timestamp": str(agent.memory_module.short_term_memory[-1]['timestamp'] 
                                      if agent.memory_module.short_term_memory else "unknown")
                    }
                    
                    self.solutions.append(system_entry)
                    
                    # Broadcast to global workspace if available
                    if self.global_workspace:
                        self.global_workspace.inject_information(
                            content={
                                "agent_name": agent.name,
                                "agent_role": agent.role,
                                "feedback": feedback,
                                "round": i + 1
                            },
                            source=agent.name,
                            activation=0.9,  # Higher activation for system agents
                            metadata={
                                "stage": "solutions",
                                "method": "PAST",
                                "agent_id": agent.agent_id,
                                "round": i + 1,
                                "system_feedback": True
                            }
                        )
                        
            logger.info(f"Completed solution round {i+1}/{depth}")
            
            # If global workspace is available, run cycles to process information
            if self.global_workspace:
                self.global_workspace.run_cycles(3)  # Run 3 cycles of global workspace
                
        return self.solutions
        
    def integrate_task_focus(self) -> Dict[str, Any]:
        """
        Focus all contributions toward the original task/question.
        
        Returns:
            Integrated final solution
        """
        if not self.original_question:
            raise ValueError("Original question not set")
            
        if not self.solutions:
            self.generate_solutions()
            
        logger.info("Integrating solutions with task focus")
        
        # Move to task stage
        self.stage = "task"
        
        # Compile all solutions
        all_solutions = ""
        for solution in self.solutions:
            # Skip system feedback for the final compilation
            if solution.get('system_feedback', False):
                continue
                
            all_solutions += f"{solution['agent_name']} ({solution['agent_role']}): {solution['solution']}\n\n"
            
        # Create integration prompt
        prompt = (
            f"Integrate the following expert contributions into a cohesive solution to the original question:\n\n"
            f"Original Question: {self.original_question}\n\n"
            f"Expert Contributions:\n{all_solutions}\n\n"
            f"Please provide a comprehensive, integrated solution that addresses the original question "
            f"effectively. Synthesize the expert contributions, eliminating redundancies and ensuring "
            f"all relevant aspects are covered. Structure the solution logically and make it directly "
            f"applicable to the original question."
        )
        
        # Use first non-system agent for integration
        integration_agent = None
        for agent in self.agents:
            if agent.name not in ["GuidingAgent", "RefinementAgent", "EvaluationAgent"]:
                integration_agent = agent
                break
                
        # If no suitable agent found, use LLM directly
        if integration_agent:
            integrated_solution = integration_agent.generate_response(prompt, temperature=0.4)
            integrator = f"{integration_agent.name} ({integration_agent.role})"
        else:
            # Fallback to direct LLM call
            messages = [{'role': 'user', 'content': prompt}]
            
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=messages,
                    options={'temperature': 0.4}
                )
                
                integrated_solution = response['message']['content'].strip()
                integrator = "System Integrator"
                
            except Exception as e:
                logger.error(f"Error integrating solutions: {e}")
                integrated_solution = "Error integrating solutions: " + str(e)
                integrator = "Error"
                
        # Store final solution
        self.final_solution = {
            "original_question": self.original_question,
            "integrated_solution": integrated_solution,
            "integrator": integrator,
            "contributors": [
                {"name": solution['agent_name'], "role": solution['agent_role']}
                for solution in self.solutions
                if not solution.get('system_feedback', False)
            ],
            "solution_count": len([s for s in self.solutions if not s.get('system_feedback', False)]),
            "stage": "task",
            "method": "PAST"
        }
        
        # Broadcast to global workspace if available
        if self.global_workspace:
            self.global_workspace.inject_information(
                content={
                    "original_question": self.original_question,
                    "integrated_solution": integrated_solution,
                    "method": "PAST"
                },
                source="PAST_method",
                activation=1.0,  # Highest activation for final solution
                metadata={
                    "stage": "task",
                    "method": "PAST",
                    "final_solution": True
                }
            )
            
            # Run global workspace cycles
            self.global_workspace.run_cycles(3)
            
        return self.final_solution
        
    def execute_full_method(self, question: str, depth: int = PAST_DEPTH) -> Dict[str, Any]:
        """
        Execute the full PAST method on a question.
        
        Args:
            question: The original question
            depth: Depth of solution generation
            
        Returns:
            Final integrated solution
        """
        logger.info(f"Executing full PAST method on question: {question[:100]}...")
        
        # Set question
        self.set_question(question)
        
        # Stage 1: Personas - Analyze question to identify required expertise
        analysis = self.analyze_question()
        
        # Stage 2: Actions - Assign actions to agents
        actions = self.assign_actions()
        
        # Stage 3: Solutions - Generate collaborative solutions
        solutions = self.generate_solutions(depth=depth)
        
        # Stage 4: Task - Integrate solutions with focus on original task
        final_solution = self.integrate_task_focus()
        
        return final_solution

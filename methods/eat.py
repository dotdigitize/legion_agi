"""
EAT (Evaluation, Action, and Testing) Method

This module implements the EAT reasoning method which focuses on practical evaluation
of proposed solutions, actionable recommendations, and testing the viability of ideas.
It represents the final iterative phase of the reasoning pipeline.
"""

import ollama
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from legion_agi.agents.agent_base import Agent
from legion_agi.core.global_workspace import GlobalWorkspace, InformationUnit
from legion_agi.utils.db_manager import DatabaseManager
from legion_agi.config import DEFAULT_MODEL, EAT_EVALUATION_THRESHOLD


class EATMethod:
    """
    Implementation of the EAT reasoning method.
    
    EAT stands for:
    - Evaluation: Critically evaluate the effectiveness of proposed solutions
    - Action: Develop specific actionable recommendations
    - Testing: Identify ways to test the viability of the proposals
    """
    
    def __init__(
        self,
        agents: List[Agent],
        input_solution: Optional[Dict[str, Any]] = None,
        global_workspace: Optional[GlobalWorkspace] = None,
        db_manager: Optional[DatabaseManager] = None,
        model: str = DEFAULT_MODEL
    ):
        """
        Initialize EAT method.
        
        Args:
            agents: List of participating agents
            input_solution: Optional input solution from previous methods
            global_workspace: Optional global workspace for coordination
            db_manager: Optional database manager for persistence
            model: LLM model for fallback reasoning
        """
        self.agents = agents
        self.input_solution = input_solution
        self.global_workspace = global_workspace
        self.db_manager = db_manager
        self.model = model
        
        # Internal state
        self.question: Optional[str] = None
        self.stage = "init"  # init, evaluation, action, testing, final
        self.evaluations: List[Dict[str, Any]] = []
        self.actions: List[Dict[str, Any]] = []
        self.tests: List[Dict[str, Any]] = []
        self.final_output: Optional[Dict[str, Any]] = None
        
        logger.info(f"EAT method initialized with {len(agents)} agents")
        
    def set_question_and_solution(self, 
                                question: str, 
                                input_solution: Dict[str, Any]) -> None:
        """
        Set the question and input solution for the EAT process.
        
        Args:
            question: The original question or problem statement
            input_solution: Input solution from previous methods
        """
        self.question = question
        self.input_solution = input_solution
        self.stage = "evaluation"
        
        # Reset state
        self.evaluations = []
        self.actions = []
        self.tests = []
        self.final_output = None
        
        logger.info(f"EAT method question and solution set: {question[:100]}...")
        
    def evaluate_solution(self) -> List[Dict[str, Any]]:
        """
        Critically evaluate the effectiveness of the proposed solution.
        
        Returns:
            List of evaluation results
        """
        if not self.question or not self.input_solution:
            raise ValueError("Question and input solution must be set")
            
        logger.info("Evaluating solution effectiveness")
        
        # Move to evaluation stage
        self.stage = "evaluation"
        
        # Extract solution content
        if isinstance(self.input_solution, dict):
            if "integrated_solution" in self.input_solution:
                solution_content = self.input_solution["integrated_solution"]
            elif "final_synthesis" in self.input_solution:
                solution_content = self.input_solution["final_synthesis"]
            else:
                solution_content = str(self.input_solution)
        else:
            solution_content = str(self.input_solution)
            
        # Each agent evaluates the solution
        for agent in self.agents:
            # System agents are often better at evaluation
            if agent.name in ["EvaluationAgent"]:
                evaluation_weight = 1.5  # Give more weight to evaluation agents
            else:
                evaluation_weight = 1.0
                
            # Create evaluation prompt
            prompt = (
                f"Critically evaluate the effectiveness of the following solution to this question:\n\n"
                f"Question: {self.question}\n\n"
                f"Proposed Solution:\n{solution_content}\n\n"
                f"As {agent.role}, evaluate this solution on the following criteria:\n"
                f"1. Correctness and accuracy\n"
                f"2. Completeness - does it address all aspects of the problem?\n"
                f"3. Practicality and feasibility\n"
                f"4. Potential unintended consequences or risks\n"
                f"5. Overall effectiveness in addressing the original question\n\n"
                f"For each criterion, provide a score from 1-10, with 10 being the highest, "
                f"and explain your reasoning. Then provide an overall score and summary assessment."
            )
            
            # Generate evaluation
            evaluation = agent.generate_response(prompt, temperature=0.3)
            
            # Try to extract overall score
            try:
                import re
                score_pattern = r"overall\s+score\s*:?\s*(\d+(?:\.\d+)?)|score\s*:?\s*(\d+(?:\.\d+)?)"
                score_match = re.search(score_pattern, evaluation.lower())
                
                if score_match:
                    score_str = score_match.group(1) or score_match.group(2)
                    overall_score = float(score_str)
                    # Normalize to 0-1 range
                    overall_score = overall_score / 10.0
                else:
                    # Default middle score if not found
                    overall_score = 0.5
            except:
                overall_score = 0.5
                
            # Apply agent weight
            weighted_score = overall_score * evaluation_weight
            
            # Store evaluation
            evaluation_entry = {
                "agent_id": agent.agent_id,
                "agent_name": agent.name,
                "agent_role": agent.role,
                "evaluation": evaluation,
                "overall_score": overall_score,
                "weighted_score": weighted_score,
                "timestamp": str(agent.memory_module.short_term_memory[-1]['timestamp'] 
                              if agent.memory_module.short_term_memory else "unknown")
            }
            
            self.evaluations.append(evaluation_entry)
            
            # Broadcast to global workspace if available
            if self.global_workspace:
                self.global_workspace.inject_information(
                    content={
                        "agent_name": agent.name,
                        "agent_role": agent.role,
                        "evaluation": evaluation,
                        "overall_score": overall_score
                    },
                    source=agent.name,
                    activation=0.8,
                    metadata={
                        "stage": "evaluation",
                        "method": "EAT",
                        "agent_id": agent.agent_id
                    }
                )
                
        return self.evaluations
        
    def propose_actions(self) -> List[Dict[str, Any]]:
        """
        Develop specific actionable recommendations based on the solution.
        
        Returns:
            List of action proposals
        """
        if not self.question or not self.input_solution:
            raise ValueError("Question and input solution must be set")
            
        if not self.evaluations:
            self.evaluate_solution()
            
        logger.info("Proposing actionable recommendations")
        
        # Move to action stage
        self.stage = "action"
        
        # Calculate average evaluation score
        if self.evaluations:
            avg_score = sum(e["weighted_score"] for e in self.evaluations) / len(self.evaluations)
        else:
            avg_score = 0.5
            
        # Extract solution content
        if isinstance(self.input_solution, dict):
            if "integrated_solution" in self.input_solution:
                solution_content = self.input_solution["integrated_solution"]
            elif "final_synthesis" in self.input_solution:
                solution_content = self.input_solution["final_synthesis"]
            else:
                solution_content = str(self.input_solution)
        else:
            solution_content = str(self.input_solution)
            
        # Format evaluations
        evaluation_summary = ""
        for i, eval_entry in enumerate(self.evaluations[:3]):  # Just include top 3 for brevity
            evaluation_summary += f"Evaluation {i+1} by {eval_entry['agent_name']} ({eval_entry['agent_role']}):\n"
            evaluation_summary += f"{eval_entry['evaluation'][:500]}...\n\n"  # Truncate for brevity
            
        # Each agent proposes actionable recommendations
        for agent in self.agents:
            # Skip pure evaluation agents, focus on domain experts
            if agent.name in ["EvaluationAgent"]:
                continue
                
            # Create action prompt
            prompt = (
                f"Based on the following solution and its evaluations, propose specific actionable "
                f"recommendations:\n\n"
                f"Question: {self.question}\n\n"
                f"Proposed Solution:\n{solution_content}\n\n"
                f"Evaluation Summary:\n{evaluation_summary}\n\n"
                f"As {agent.role}, propose 3-5 specific, actionable recommendations that would:\n"
                f"1. Implement the key elements of the solution\n"
                f"2. Address any weaknesses identified in the evaluations\n"
                f"3. Maximize the solution's effectiveness\n"
                f"4. Mitigate potential risks or challenges\n\n"
                f"For each recommendation, provide:\n"
                f"- A clear, specific action to take\n"
                f"- The rationale for this action\n"
                f"- Expected outcome or impact\n"
                f"- Any prerequisites or dependencies\n\n"
                f"Focus on concrete, implementable actions rather than general advice."
            )
            
            # Generate action recommendations
            action_proposals = agent.generate_response(prompt, temperature=0.5)
            
            # Store action proposals
            action_entry = {
                "agent_id": agent.agent_id,
                "agent_name": agent.name,
                "agent_role": agent.role,
                "action_proposals": action_proposals,
                "evaluation_score": avg_score,
                "timestamp": str(agent.memory_module.short_term_memory[-1]['timestamp'] 
                              if agent.memory_module.short_term_memory else "unknown")
            }
            
            self.actions.append(action_entry)
            
            # Broadcast to global workspace if available
            if self.global_workspace:
                self.global_workspace.inject_information(
                    content={
                        "agent_name": agent.name,
                        "agent_role": agent.role,
                        "action_proposals": action_proposals
                    },
                    source=agent.name,
                    activation=0.8,
                    metadata={
                        "stage": "action",
                        "method": "EAT",
                        "agent_id": agent.agent_id
                    }
                )
                
        return self.actions
        
    def propose_tests(self) -> List[Dict[str, Any]]:
        """
        Identify ways to test the viability of the proposed solution and actions.
        
        Returns:
            List of test proposals
        """
        if not self.question or not self.input_solution:
            raise ValueError("Question and input solution must be set")
            
        if not self.actions:
            self.propose_actions()
            
        logger.info("Proposing tests for solution viability")
        
        # Move to testing stage
        self.stage = "testing"
        
        # Format action proposals
        action_summary = ""
        for i, action_entry in enumerate(self.actions[:3]):  # Just include top 3 for brevity
            action_summary += f"Actions proposed by {action_entry['agent_name']} ({action_entry['agent_role']}):\n"
            action_summary += f"{action_entry['action_proposals'][:500]}...\n\n"  # Truncate for brevity
            
        # Each agent proposes tests
        for agent in self.agents:
            # System agents and domain experts can both provide valuable test perspectives
            if agent.name in ["EvaluationAgent", "RefinementAgent"]:
                # Give more weight to system agents for testing
                test_weight = 1.5
            else:
                test_weight = 1.0
                
            # Create test prompt
            prompt = (
                f"Propose specific tests to verify the viability and effectiveness of the following "
                f"solution and action recommendations:\n\n"
                f"Question: {self.question}\n\n"
                f"Action Recommendations Summary:\n{action_summary}\n\n"
                f"As {agent.role}, propose 3-5 specific tests that would:\n"
                f"1. Validate the underlying assumptions\n"
                f"2. Verify the effectiveness of the proposed actions\n"
                f"3. Identify potential weaknesses or failure modes\n"
                f"4. Measure success or impact\n\n"
                f"For each test, specify:\n"
                f"- The specific hypothesis or aspect being tested\n"
                f"- Testing methodology and process\n"
                f"- Success criteria and metrics\n"
                f"- Resources needed and estimated timeframe\n"
                f"- How to interpret the results\n\n"
                f"Focus on practical, feasible tests that provide meaningful validation."
            )
            
            # Generate test proposals
            test_proposals = agent.generate_response(prompt, temperature=0.4)
            
            # Store test proposals
            test_entry = {
                "agent_id": agent.agent_id,
                "agent_name": agent.name,
                "agent_role": agent.role,
                "test_proposals": test_proposals,
                "test_weight": test_weight,
                "timestamp": str(agent.memory_module.short_term_memory[-1]['timestamp'] 
                              if agent.memory_module.short_term_memory else "unknown")
            }
            
            self.tests.append(test_entry)
            
            # Broadcast to global workspace if available
            if self.global_workspace:
                self.global_workspace.inject_information(
                    content={
                        "agent_name": agent.name,
                        "agent_role": agent.role,
                        "test_proposals": test_proposals
                    },
                    source=agent.name,
                    activation=0.8,
                    metadata={
                        "stage": "testing",
                        "method": "EAT",
                        "agent_id": agent.agent_id
                    }
                )
                
        return self.tests
        
    def compile_final_output(self) -> Dict[str, Any]:
        """
        Compile the final output with solution, actions, and tests.
        
        Returns:
            Final compiled output
        """
        if not self.question or not self.input_solution:
            raise ValueError("Question and input solution must be set")
            
        if not self.tests:
            self.propose_tests()
            
        logger.info("Compiling final EAT output")
        
        # Move to final stage
        self.stage = "final"
        
        # Calculate average evaluation score
        if self.evaluations:
            avg_score = sum(e["weighted_score"] for e in self.evaluations) / len(self.evaluations)
        else:
            avg_score = 0.5
            
        # Check if solution meets the evaluation threshold
        solution_acceptable = avg_score >= EAT_EVALUATION_THRESHOLD
        
        # Extract solution content
        if isinstance(self.input_solution, dict):
            if "integrated_solution" in self.input_solution:
                solution_content = self.input_solution["integrated_solution"]
            elif "final_synthesis" in self.input_solution:
                solution_content = self.input_solution["final_synthesis"]
            else:
                solution_content = str(self.input_solution)
        else:
            solution_content = str(self.input_solution)
            
        # Compile action recommendations
        action_recommendations = []
        for action_entry in self.actions:
            action_recommendations.append({
                "agent_name": action_entry["agent_name"],
                "agent_role": action_entry["agent_role"],
                "recommendations": action_entry["action_proposals"]
            })
            
        # Compile test proposals
        test_recommendations = []
        for test_entry in self.tests:
            test_recommendations.append({
                "agent_name": test_entry["agent_name"],
                "agent_role": test_entry["agent_role"],
                "test_proposals": test_entry["test_proposals"],
                "weight": test_entry["test_weight"]
            })
            
        # Compile final output
        self.final_output = {
            "question": self.question,
            "solution": solution_content,
            "evaluation_score": avg_score,
            "solution_acceptable": solution_acceptable,
            "action_recommendations": action_recommendations,
            "test_recommendations": test_recommendations,
            "stage": "final",
            "method": "EAT"
        }
        
        # Create executive summary
        summary_prompt = (
            f"Create a concise executive summary of the following solution evaluation, "
            f"actions, and tests:\n\n"
            f"Question: {self.question}\n\n"
            f"Solution Evaluation Score: {avg_score*10}/10\n"
            f"Action Recommendations: {len(action_recommendations)}\n"
            f"Test Proposals: {len(test_recommendations)}\n\n"
            f"Solution Acceptable: {'Yes' if solution_acceptable else 'No'}\n\n"
            f"Write a 3-5 paragraph executive summary that highlights the key points of the solution, "
            f"most important actions to take, and critical tests to perform. Be concise and focus on "
            f"the most important aspects that a decision-maker would need to know."
        )
        
        # Generate executive summary
        messages = [{'role': 'user', 'content': summary_prompt}]
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={'temperature': 0.3}
            )
            
            executive_summary = response['message']['content'].strip()
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            executive_summary = f"Error generating executive summary: {e}"
            
        # Add executive summary to final output
        self.final_output["executive_summary"] = executive_summary
        
        # Broadcast to global workspace if available
        if self.global_workspace:
            self.global_workspace.inject_information(
                content={
                    "question": self.question,
                    "executive_summary": executive_summary,
                    "evaluation_score": avg_score,
                    "solution_acceptable": solution_acceptable,
                    "method": "EAT"
                },
                source="EAT_method",
                activation=1.0,  # Highest activation for final output
                metadata={
                    "stage": "final",
                    "method": "EAT",
                    "final_output": True
                }
            )
            
            # Run global workspace cycles
            self.global_workspace.run_cycles(3)
            
        return self.final_output
        
    def execute_full_method(self, 
                          question: str, 
                          input_solution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the full EAT method on a question and input solution.
        
        Args:
            question: The original question or problem statement
            input_solution: Input solution from previous methods
            
        Returns:
            Final compiled output
        """
        logger.info(f"Executing full EAT method on question: {question[:100]}...")
        
        # Set question and input solution
        self.set_question_and_solution(question, input_solution)
        
        # Stage 1: Evaluation - Evaluate solution effectiveness
        evaluations = self.evaluate_solution()
        
        # Stage 2: Action - Propose actionable recommendations
        actions = self.propose_actions()
        
        # Stage 3: Testing - Propose tests for solution viability
        tests = self.propose_tests()
        
        # Compile final output
        final_output = self.compile_final_output()
        
        return final_output

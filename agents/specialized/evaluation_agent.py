"""
Evaluation Agent for Legion AGI

This specialized agent focuses on evaluating solutions for effectiveness, 
feasibility, coherence, and alignment with the original problem.
"""

from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from legion_agi.agents.agent_base import Agent
from legion_agi.core.global_workspace import GlobalWorkspace
from legion_agi.core.memory_module import MemoryModule
from legion_agi.utils.db_manager import DatabaseManager


class EvaluationAgent(Agent):
    """
    Specialized agent for evaluating the quality, feasibility, and effectiveness of solutions.
    Uses structured evaluation frameworks to provide objective assessment.
    """
    
    def __init__(
        self,
        name: str,
        role: str,
        backstory: str,
        style: str,
        instructions: str,
        model: str = "llama3.1:8b",
        db_manager: Optional[DatabaseManager] = None,
        memory_module: Optional[MemoryModule] = None,
        global_workspace: Optional[GlobalWorkspace] = None
    ):
        """Initialize Evaluation Agent with specialized parameters."""
        super().__init__(
            name=name,
            role=role,
            backstory=backstory,
            style=style,
            instructions=instructions,
            model=model,
            db_manager=db_manager,
            memory_module=memory_module,
            global_workspace=global_workspace
        )
        
        # Evaluation-specific attributes
        self.evaluation_history = []
        self.evaluation_frameworks = self._initialize_evaluation_frameworks()
        
        # Cognitive parameters optimized for evaluation
        self.attention_span = 0.95  # Very high attention to detail
        self.creativity = 0.4       # Lower creativity for objective evaluation
        
        logger.info(f"Evaluation Agent '{name}' initialized")
        
    def _initialize_evaluation_frameworks(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize evaluation frameworks for different types of problems.
        
        Returns:
            Dictionary of evaluation frameworks
        """
        frameworks = {
            "general": {
                "criteria": [
                    {"name": "correctness", "weight": 0.25, "description": "Accuracy and correctness of the solution"},
                    {"name": "completeness", "weight": 0.20, "description": "Addresses all aspects of the problem"},
                    {"name": "feasibility", "weight": 0.20, "description": "Practical implementation potential"},
                    {"name": "clarity", "weight": 0.15, "description": "Clear, understandable explanation"},
                    {"name": "innovation", "weight": 0.10, "description": "Novel or creative approach"},
                    {"name": "efficiency", "weight": 0.10, "description": "Optimal use of resources"}
                ],
                "scale": [
                    {"score": 1, "label": "Poor", "description": "Significant issues or inadequacies"},
                    {"score": 2, "label": "Fair", "description": "Notable weaknesses but some merit"},
                    {"score": 3, "label": "Good", "description": "Solid with minor weaknesses"},
                    {"score": 4, "label": "Very Good", "description": "Strong with minimal issues"},
                    {"score": 5, "label": "Excellent", "description": "Outstanding with no significant flaws"}
                ]
            },
            "technical": {
                "criteria": [
                    {"name": "correctness", "weight": 0.25, "description": "Technical accuracy and correctness"},
                    {"name": "scalability", "weight": 0.20, "description": "Ability to scale to larger problems"},
                    {"name": "efficiency", "weight": 0.20, "description": "Computational or resource efficiency"},
                    {"name": "robustness", "weight": 0.15, "description": "Handles edge cases and failures"},
                    {"name": "maintainability", "weight": 0.10, "description": "Ease of maintenance and extension"},
                    {"name": "security", "weight": 0.10, "description": "Security considerations and safeguards"}
                ],
                "scale": [
                    {"score": 1, "label": "Poor", "description": "Significant technical issues"},
                    {"score": 2, "label": "Fair", "description": "Notable technical weaknesses"},
                    {"score": 3, "label": "Good", "description": "Technically sound with minor issues"},
                    {"score": 4, "label": "Very Good", "description": "Strong technical solution with minimal issues"},
                    {"score": 5, "label": "Excellent", "description": "Technically outstanding solution"}
                ]
            },
            "research": {
                "criteria": [
                    {"name": "methodology", "weight": 0.25, "description": "Sound research methodology"},
                    {"name": "evidence", "weight": 0.20, "description": "Quality and relevance of evidence"},
                    {"name": "analysis", "weight": 0.20, "description": "Depth and rigor of analysis"},
                    {"name": "novelty", "weight": 0.15, "description": "Originality of contribution"},
                    {"name": "implications", "weight": 0.10, "description": "Significance of implications"},
                    {"name": "limitations", "weight": 0.10, "description": "Acknowledgment of limitations"}
                ],
                "scale": [
                    {"score": 1, "label": "Poor", "description": "Significant methodological issues"},
                    {"score": 2, "label": "Fair", "description": "Notable weaknesses in research approach"},
                    {"score": 3, "label": "Good", "description": "Solid research with minor limitations"},
                    {"score": 4, "label": "Very Good", "description": "Strong research with minimal issues"},
                    {"score": 5, "label": "Excellent", "description": "Outstanding research contribution"}
                ]
            }
        }
        
        return frameworks
        
    def respond(
        self,
        conversation_history: List[Dict[str, str]],
        original_question: str,
        framework_type: str = "general",
        temperature: float = 0.4,
        top_p: float = 0.8
    ) -> str:
        """
        Generate an evaluation response based on conversation history.
        
        Args:
            conversation_history: History of the conversation
            original_question: The original question/problem to solve
            framework_type: Type of evaluation framework to use
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Evaluation feedback
        """
        # Format conversation history
        formatted_history = self.format_conversation_history(conversation_history)
        
        # Extract solutions to evaluate
        solutions_to_evaluate = self._extract_solutions(conversation_history)
        
        # Get appropriate evaluation framework
        framework = self.evaluation_frameworks.get(
            framework_type, 
            self.evaluation_frameworks["general"]
        )
        
        # Format framework for prompt
        criteria_str = "\n".join([
            f"- {c['name'].capitalize()} ({int(c['weight']*100)}%): {c['description']}"
            for c in framework["criteria"]
        ])
        
        scale_str = "\n".join([
            f"- {s['score']} ({s['label']}): {s['description']}"
            for s in framework["scale"]
        ])
        
        # Build evaluation prompt
        prompt = (
            f"You are {self.name}, {self.role}.\n"
            f"Backstory: {self.backstory}\n"
            f"Communication Style: {self.style}\n"
            f"{self.instructions}\n\n"
            f"Your task is to evaluate the proposed solutions to the following problem:\n\n"
            f"Original Question: {original_question}\n\n"
            f"Conversation History and Proposed Solutions:\n{formatted_history}\n\n"
            f"Please evaluate these solutions using the following framework:\n\n"
            f"Evaluation Criteria:\n{criteria_str}\n\n"
            f"Rating Scale (1-5):\n{scale_str}\n\n"
            f"For your evaluation:\n"
            f"1. Assess each criterion objectively\n"
            f"2. Provide specific evidence from the solutions to justify your ratings\n"
            f"3. Identify key strengths and weaknesses\n"
            f"4. Suggest concrete improvements\n"
            f"5. Provide an overall assessment and recommendation\n\n"
            f"Structure your evaluation clearly, focusing on objective assessment."
        )
        
        # Generate evaluation
        response_content = self.generate_response(
            prompt, 
            temperature=temperature,
            top_p=top_p
        )
        
        # Save to conversation history if DB is available
        if self.db_manager:
            self.db_manager.save_conversation(self.agent_id, self.name, response_content)
            
        # Add to memory
        self.memory_module.add_to_stm(
            content=response_content,
            source="self",
            metadata={
                "type": "evaluation",
                "original_question": original_question,
                "framework_type": framework_type,
                "solutions_evaluated": len(solutions_to_evaluate)
            },
            importance=0.85  # High importance for evaluations
        )
        
        # Save to evaluation history
        evaluation_entry = {
            "original_question": original_question,
            "framework_type": framework_type,
            "solutions_evaluated": solutions_to_evaluate,
            "evaluation": response_content,
            "timestamp": str(self.memory_module.short_term_memory[-1]['timestamp']
                           if self.memory_module.short_term_memory else "unknown")
        }
        self.evaluation_history.append(evaluation_entry)
        
        # Broadcast to global workspace if available
        if self.global_workspace:
            self.global_workspace.inject_information(
                content={
                    "agent_name": self.name,
                    "evaluation": response_content,
                    "original_question": original_question,
                    "framework_type": framework_type
                },
                source=self.name,
                activation=0.9,  # Very high activation for evaluations
                metadata={
                    "type": "evaluation",
                    "solutions_evaluated": len(solutions_to_evaluate)
                }
            )
            
        return response_content
        
    def _extract_solutions(self, conversation_history: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Extract solutions from conversation history for evaluation.
        
        Args:
            conversation_history: Conversation history
            
        Returns:
            List of extracted solutions with metadata
        """
        solutions = []
        
        for entry in conversation_history:
            role = entry.get('role', '')
            content = entry.get('content', '')
            
            # Skip entries from system agents
            if role in ["GuidingAgent", "EvaluationAgent", "System"]:
                continue
                
            # Include RefinementAgent entries as they contain refined solutions
            if role == "RefinementAgent" and len(content) > 200:
                solutions.append({
                    "role": role,
                    "content": content,
                    "length": len(content),
                    "is_refinement": True
                })
                continue
                
            # Skip very short entries (unlikely to be solutions)
            if len(content) < 150:
                continue
                
            # Add to solutions list
            solutions.append({
                "role": role,
                "content": content,
                "length": len(content),
                "is_refinement": False
            })
            
        return solutions
    
    def evaluate_specific_solution(self, 
                                 solution: str, 
                                 problem: str,
                                 framework_type: str = "general") -> Dict[str, Any]:
        """
        Evaluate a specific solution using structured criteria.
        
        Args:
            solution: The solution to evaluate
            problem: The problem statement
            framework_type: Type of evaluation framework to use
            
        Returns:
            Structured evaluation results
        """
        # Get appropriate framework
        framework = self.evaluation_frameworks.get(
            framework_type, 
            self.evaluation_frameworks["general"]
        )
        
        # Format framework for prompt
        criteria_str = "\n".join([
            f"- {c['name'].capitalize()} ({int(c['weight']*100)}%): {c['description']}"
            for c in framework["criteria"]
        ])
        
        scale_str = "\n".join([
            f"- {s['score']} ({s['label']}): {s['description']}"
            for s in framework["scale"]
        ])
        
        # Build evaluation prompt
        prompt = (
            f"You are {self.name}, {self.role}.\n"
            f"Backstory: {self.backstory}\n"
            f"Communication Style: {self.style}\n"
            f"{self.instructions}\n\n"
            f"Please evaluate the following solution to the problem:\n\n"
            f"Problem: {problem}\n\n"
            f"Solution to Evaluate:\n{solution}\n\n"
            f"Evaluation Framework:\n\n"
            f"Criteria:\n{criteria_str}\n\n"
            f"Rating Scale (1-5):\n{scale_str}\n\n"
            f"For each criterion, provide:\n"
            f"1. A numerical score (1-5)\n"
            f"2. A brief justification for the score\n"
            f"3. Specific strengths or weaknesses\n\n"
            f"Conclude with an overall weighted score and summary assessment.\n\n"
            f"Format your response as a structured evaluation with clear sections for each criterion."
        )
        
        # Generate structured evaluation
        evaluation = self.generate_response(prompt, temperature=0.3)
        
        # Try to extract scores from the evaluation
        scores = self._extract_scores_from_evaluation(evaluation, framework)
        
        # Create structured result
        result = {
            "problem": problem,
            "solution_length": len(solution),
            "framework_type": framework_type,
            "full_evaluation": evaluation,
            "scores": scores,
            "timestamp": str(self.memory_module.short_term_memory[-1]['timestamp']
                           if self.memory_module.short_term_memory else "unknown")
        }
        
        # Add to memory
        self.memory_module.add_to_stm(
            content=evaluation,
            source="self",
            metadata={
                "type": "specific_evaluation",
                "framework_type": framework_type,
                "scores": scores
            },
            importance=0.8
        )
        
        return result
        
    def _extract_scores_from_evaluation(self, 
                                      evaluation: str, 
                                      framework: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract numerical scores from evaluation text.
        
        Args:
            evaluation: Evaluation text
            framework: Evaluation framework
            
        Returns:
            Dictionary of criterion scores
        """
        scores = {}
        
        # For each criterion, try to find the score
        for criterion in framework["criteria"]:
            criterion_name = criterion["name"]
            
            # Build patterns to match against
            patterns = [
                f"{criterion_name.capitalize()}: ?([1-5])(/5)?",
                f"{criterion_name.capitalize()} score: ?([1-5])(/5)?",
                f"{criterion_name.capitalize()}: ([1-5])(/5)? -",
                f"{criterion_name.capitalize()}: ([1-5])(/5)? out of",
                f"score of ([1-5])(/5)? for {criterion_name}"
            ]
            
            # Try each pattern
            import re
            score = None
            
            for pattern in patterns:
                match = re.search(pattern, evaluation, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    break
                    
            # If found, add to scores
            if score is not None:
                scores[criterion_name] = score
                
        # Try to extract overall score
        overall_patterns = [
            r"Overall score: ?([0-9.]+)(/5)?",
            r"Overall assessment: ?([0-9.]+)(/5)?",
            r"Overall: ?([0-9.]+)(/5)?",
            r"Weighted score: ?([0-9.]+)(/5)?",
            r"Final score: ?([0-9.]+)(/5)?"
        ]
        
        for pattern in overall_patterns:
            match = re.search(pattern, evaluation, re.IGNORECASE)
            if match:
                scores["overall"] = float(match.group(1))
                break
                
        # If overall not found but we have individual scores, calculate weighted average
        if "overall" not in scores and scores:
            weighted_sum = 0
            total_weight = 0
            
            for criterion in framework["criteria"]:
                name = criterion["name"]
                weight = criterion["weight"]
                
                if name in scores:
                    weighted_sum += scores[name] * weight
                    total_weight += weight
                    
            if total_weight > 0:
                scores["overall"] = round(weighted_sum / total_weight, 2)
                
        return scores
        
    def comparison_evaluation(self, 
                            solutions: List[str], 
                            problem: str,
                            framework_type: str = "general") -> Dict[str, Any]:
        """
        Compare multiple solutions and evaluate them against each other.
        
        Args:
            solutions: List of solutions to compare
            problem: The problem statement
            framework_type: Type of evaluation framework to use
            
        Returns:
            Comparative evaluation results
        """
        if len(solutions) < 2:
            return {"error": "Need at least 2 solutions for comparison"}
            
        # Get appropriate framework
        framework = self.evaluation_frameworks.get(
            framework_type, 
            self.evaluation_frameworks["general"]
        )
        
        # Format solutions for prompt
        solutions_str = ""
        for i, solution in enumerate(solutions):
            solutions_str += f"Solution {i+1}:\n{solution}\n\n"
            
        # Format framework
        criteria_str = "\n".join([
            f"- {c['name'].capitalize()}: {c['description']}"
            for c in framework["criteria"]
        ])
        
        # Build comparison prompt
        prompt = (
            f"You are {self.name}, {self.role}.\n"
            f"Backstory: {self.backstory}\n"
            f"Communication Style: {self.style}\n"
            f"{self.instructions}\n\n"
            f"Please compare and evaluate the following solutions to the problem:\n\n"
            f"Problem: {problem}\n\n"
            f"{solutions_str}\n"
            f"Compare these solutions using the following criteria:\n{criteria_str}\n\n"
            f"For your evaluation:\n"
            f"1. Compare the solutions side-by-side for each criterion\n"
            f"2. Identify the relative strengths and weaknesses of each solution\n"
            f"3. Determine which solution is strongest for each criterion\n"
            f"4. Suggest how elements from different solutions could be combined for an optimal approach\n"
            f"5. Provide an overall recommendation on which solution is best and why\n\n"
            f"Structure your comparison clearly, with separate sections for each criterion and a final recommendation."
        )
        
        # Generate comparison
        comparison = self.generate_response(prompt, temperature=0.4)
        
        # Try to determine best solution
        best_solution_idx = self._extract_best_solution(comparison, len(solutions))
        
        # Create result
        result = {
            "problem": problem,
            "num_solutions": len(solutions),
            "framework_type": framework_type,
            "comparison": comparison,
            "best_solution_index": best_solution_idx,
            "best_solution": best_solution_idx is not None,
            "timestamp": str(self.memory_module.short_term_memory[-1]['timestamp']
                           if self.memory_module.short_term_memory else "unknown")
        }
        
        # Add to memory
        self.memory_module.add_to_stm(
            content=comparison,
            source="self",
            metadata={
                "type": "comparison_evaluation",
                "framework_type": framework_type,
                "num_solutions": len(solutions),
                "best_solution_index": best_solution_idx
            },
            importance=0.85
        )
        
        return result
        
    def _extract_best_solution(self, comparison: str, num_solutions: int) -> Optional[int]:
        """
        Extract the index of the best solution from comparison text.
        
        Args:
            comparison: Comparison evaluation text
            num_solutions: Number of solutions compared
            
        Returns:
            Index of best solution (0-based) or None if not determined
        """
        # Patterns to match statements about best solution
        import re
        patterns = [
            r"Solution ([0-9]+) is the best",
            r"Solution ([0-9]+) offers the strongest",
            r"Solution ([0-9]+) provides the most",
            r"Solution ([0-9]+) is recommended",
            r"recommend Solution ([0-9]+)",
            r"prefer Solution ([0-9]+)",
            r"Solution ([0-9]+) is superior",
            r"Solution ([0-9]+) outperforms"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, comparison, re.IGNORECASE)
            if match:
                try:
                    solution_num = int(match.group(1))
                    if 1 <= solution_num <= num_solutions:
                        return solution_num - 1  # Convert to 0-based index
                except:
                    pass
                    
        return None

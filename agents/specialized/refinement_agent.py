"""
Refinement Agent for Legion AGI

This specialized agent focuses on refining ideas and solutions proposed by other agents.
It identifies gaps, inconsistencies, and areas for improvement in collaborative solutions.
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from legion_agi.agents.agent_base import Agent
from legion_agi.core.global_workspace import GlobalWorkspace
from legion_agi.core.memory_module import MemoryModule
from legion_agi.utils.db_manager import DatabaseManager


class RefinementAgent(Agent):
    """
    Specialized agent for refining solutions through structured feedback and critical analysis.
    Takes existing solutions and identifies improvements, gaps, and inconsistencies.
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
        """Initialize Refinement Agent with specialized parameters."""
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
        
        # Refinement-specific attributes
        self.refinement_history = []
        self.current_context = {}
        
        # Enhanced attention to detail for refinement tasks
        self.attention_span = 0.9  # High attention to detail
        self.creativity = 0.6      # Moderate creativity for refinement
        
        logger.info(f"Refinement Agent '{name}' initialized")
        
    def respond(
        self,
        conversation_history: List[Dict[str, str]],
        original_question: str,
        temperature: float = 0.6,
        top_p: float = 0.8
    ) -> str:
        """
        Generate a refinement response based on conversation history.
        
        Args:
            conversation_history: History of the conversation
            original_question: The original question/problem to solve
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Refinement feedback
        """
        # Store original question for context
        self.current_context["original_question"] = original_question
        
        # Format conversation history
        formatted_history = self.format_conversation_history(conversation_history)
        
        # Identify contributions to refine
        solutions_to_refine = self._extract_solutions(conversation_history)
        
        # Build refinement prompt
        prompt = (
            f"You are {self.name}, {self.role}.\n"
            f"Backstory: {self.backstory}\n"
            f"Communication Style: {self.style}\n"
            f"{self.instructions}\n"
            f"Your task is to carefully review and refine the existing solutions by providing feedback, "
            f"identifying potential improvements, and suggesting actionable enhancements.\n\n"
            f"Original Question: {original_question}\n\n"
            f"Conversation History and Proposed Solutions:\n{formatted_history}\n\n"
            f"Please provide your refinement feedback. Focus on:\n"
            f"1. Identifying gaps or inconsistencies in the current solutions\n"
            f"2. Suggesting specific improvements or extensions\n"
            f"3. Evaluating the completeness and effectiveness of the solutions\n"
            f"4. Proposing concrete, actionable refinements\n\n"
            f"Structure your feedback clearly, focusing on constructive improvements."
        )
        
        # Generate refinement feedback
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
                "type": "refinement",
                "original_question": original_question,
                "solutions_refined": len(solutions_to_refine)
            },
            importance=0.8  # Higher importance for refinements
        )
        
        # Save to refinement history
        refinement_entry = {
            "original_question": original_question,
            "solutions_refined": solutions_to_refine,
            "refinement_feedback": response_content,
            "timestamp": str(self.memory_module.short_term_memory[-1]['timestamp']
                           if self.memory_module.short_term_memory else "unknown")
        }
        self.refinement_history.append(refinement_entry)
        
        # Broadcast to global workspace if available
        if self.global_workspace:
            self.global_workspace.inject_information(
                content={
                    "agent_name": self.name,
                    "refinement_feedback": response_content,
                    "original_question": original_question
                },
                source=self.name,
                activation=0.85,  # High activation for refinements
                metadata={
                    "type": "refinement",
                    "solutions_refined": len(solutions_to_refine)
                }
            )
            
        return response_content
        
    def _extract_solutions(self, conversation_history: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Extract solutions from conversation history for refinement.
        
        Args:
            conversation_history: Conversation history
            
        Returns:
            List of extracted solutions with metadata
        """
        solutions = []
        
        for entry in conversation_history:
            role = entry.get('role', '')
            content = entry.get('content', '')
            
            # Skip entries from system agents and non-solution entries
            if role in ["GuidingAgent", "RefinementAgent", "EvaluationAgent", "System"]:
                continue
                
            # Skip very short entries (unlikely to be solutions)
            if len(content) < 100:
                continue
                
            # Add to solutions list
            solutions.append({
                "role": role,
                "content": content,
                "length": len(content)
            })
            
        return solutions
    
    def provide_targeted_refinement(self, specific_solution: str, aspect: str) -> str:
        """
        Provide targeted refinement for a specific aspect of a solution.
        
        Args:
            specific_solution: The solution to refine
            aspect: Specific aspect to focus on (e.g., "completeness", "clarity", "technical feasibility")
            
        Returns:
            Targeted refinement feedback
        """
        # Build targeted refinement prompt
        prompt = (
            f"You are {self.name}, {self.role}.\n"
            f"Backstory: {self.backstory}\n"
            f"Communication Style: {self.style}\n"
            f"{self.instructions}\n\n"
            f"Original Question: {self.current_context.get('original_question', 'Unknown question')}\n\n"
            f"You need to provide targeted refinement feedback on the following solution, "
            f"focusing specifically on the aspect of {aspect}:\n\n"
            f"Solution to Refine:\n{specific_solution}\n\n"
            f"Please provide detailed, constructive feedback to improve the {aspect} of this solution. "
            f"Suggest specific changes, additions, or clarifications that would enhance this aspect."
        )
        
        # Generate targeted refinement
        targeted_feedback = self.generate_response(prompt, temperature=0.5)
        
        # Store in memory
        self.memory_module.add_to_stm(
            content=targeted_feedback,
            source="self",
            metadata={
                "type": "targeted_refinement",
                "aspect": aspect,
                "solution_length": len(specific_solution)
            },
            importance=0.75
        )
        
        return targeted_feedback
        
    def integrate_refinements(self, original_solution: str, refinements: List[str]) -> str:
        """
        Integrate multiple refinements into the original solution.
        
        Args:
            original_solution: The original solution
            refinements: List of refinement suggestions
            
        Returns:
            Integrated refined solution
        """
        # Combine refinements
        combined_refinements = "\n\n".join([f"Refinement {i+1}: {r}" for i, r in enumerate(refinements)])
        
        # Build integration prompt
        prompt = (
            f"You are {self.name}, {self.role}.\n"
            f"Backstory: {self.backstory}\n"
            f"Communication Style: {self.style}\n"
            f"{self.instructions}\n\n"
            f"Original Question: {self.current_context.get('original_question', 'Unknown question')}\n\n"
            f"Original Solution:\n{original_solution}\n\n"
            f"Refinement Suggestions:\n{combined_refinements}\n\n"
            f"Please integrate these refinements into the original solution, creating a comprehensive "
            f"improved version. Ensure the refinements are seamlessly incorporated while maintaining "
            f"the core strengths of the original solution."
        )
        
        # Generate integrated solution
        integrated_solution = self.generate_response(prompt, temperature=0.5)
        
        # Store in memory
        self.memory_module.add_to_stm(
            content=integrated_solution,
            source="self",
            metadata={
                "type": "integrated_refinement",
                "num_refinements": len(refinements)
            },
            importance=0.9  # High importance for integrated refinements
        )
        
        return integrated_solution

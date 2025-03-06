"""
Guiding Agent for Legion AGI System

This module implements the specialized GuidingAgent class that keeps the conversation
focused on the original question and helps prevent deviation from the main topic.
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from legion_agi.agents.agent_base import Agent
from legion_agi.core.global_workspace import GlobalWorkspace
from legion_agi.utils.db_manager import DatabaseManager
from legion_agi.config import DEFAULT_MODEL


class GuidingAgent(Agent):
    """
    GuidingAgent keeps the conversation focused on the original question.
    It monitors the discussion and provides gentle reminders if agents stray off-topic.
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
        global_workspace: Optional[GlobalWorkspace] = None
    ):
        """
        Initialize guiding agent.
        
        Args:
            name: Agent name
            role: Agent's role/specialty
            backstory: Agent's backstory/context
            style: Communication style
            instructions: Specific instructions for this agent
            model: LLM model to use
            db_manager: Database manager instance
            global_workspace: Global workspace instance
        """
        super().__init__(
            name=name,
            role=role,
            backstory=backstory,
            style=style,
            instructions=instructions,
            model=model,
            db_manager=db_manager,
            global_workspace=global_workspace
        )
        
        # GuidingAgent specific state
        self.original_question: Optional[str] = None
        self.topic_keywords: List[str] = []
        self.intervention_frequency = 0.3  # Only intervene when necessary (~30% of cases)
        self.last_intervention_time = 0
        self.intervention_count = 0
        
        logger.info(f"GuidingAgent '{name}' initialized")
        
    def set_original_question(self, question: str) -> None:
        """
        Set the original question to guide the conversation toward.
        
        Args:
            question: Original question text
        """
        self.original_question = question
        
        # Extract keywords from the question
        self._extract_keywords(question)
        
        # Reset intervention state
        self.last_intervention_time = 0
        self.intervention_count = 0
        
        logger.info(f"GuidingAgent set original question: {question[:100]}...")
        
    def _extract_keywords(self, text: str) -> None:
        """
        Extract important keywords from the text.
        
        Args:
            text: Text to extract keywords from
        """
        # Simple extraction based on word frequency
        # In a more sophisticated implementation, this could use NLP techniques
        words = text.lower().split()
        stopwords = {
            "the", "a", "an", "is", "are", "and", "or", "but", "in", "on", "at", 
            "to", "for", "with", "about", "from", "as", "of", "by", "that", "this",
            "these", "those", "it", "they", "them", "their", "there", "here", "who",
            "what", "when", "where", "why", "how", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "can", "could", "will", "would", "shall", "should",
            "may", "might", "must", "i", "you", "he", "she", "we", "they"
        }
        
        # Count word frequency
        word_freq = {}
        for word in words:
            # Clean word of punctuation
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word and clean_word not in stopwords and len(clean_word) > 3:
                word_freq[clean_word] = word_freq.get(clean_word, 0) + 1
                
        # Select top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        self.topic_keywords = [word for word, freq in sorted_words[:10]]
        
        logger.debug(f"Extracted keywords: {self.topic_keywords}")
        
    def assess_topic_relevance(self, content: str) -> float:
        """
        Assess how relevant content is to the original topic.
        
        Args:
            content: Content to assess
            
        Returns:
            Relevance score between 0 and 1
        """
        if not self.original_question or not self.topic_keywords:
            return 1.0  # Assume relevant if no question set
            
        # Simple keyword-based assessment
        content_lower = content.lower()
        keyword_hits = sum(1 for keyword in self.topic_keywords if keyword in content_lower)
        
        # Scale by number of keywords
        relevance = keyword_hits / max(1, min(len(self.topic_keywords), 5))
        
        # Cap between 0 and 1
        relevance = max(0.0, min(1.0, relevance))
        
        return relevance
        
    def should_intervene(self, content: str, current_time: int) -> bool:
        """
        Determine if the agent should intervene to guide the conversation.
        
        Args:
            content: Current conversation content
            current_time: Current conversation time (arbitrary units)
            
        Returns:
            True if the agent should intervene, False otherwise
        """
        # Check how much time has passed since last intervention
        time_since_last = current_time - self.last_intervention_time
        
        # Don't intervene too frequently
        if time_since_last < 3:
            return False
            
        # Assess relevance of content
        relevance = self.assess_topic_relevance(content)
        
        # Decide whether to intervene based on relevance and frequency
        should_intervene = (
            relevance < 0.4  # Content isn't very relevant
            and self.intervention_count < 5  # Don't intervene too many times total
            and time_since_last >= 3  # Ensure some time between interventions
        )
        
        return should_intervene
        
    def respond(self,
              conversation_history: List[Dict[str, str]],
              original_question: Optional[str] = None,
              temperature: float = 0.5,
              top_p: float = 0.7) -> str:
        """
        Generate a guiding response based on the conversation history.
        
        Args:
            conversation_history: History of the conversation
            original_question: Original question to guide toward
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated guiding response
        """
        # If original_question is provided, update internal state
        if original_question and original_question != self.original_question:
            self.set_original_question(original_question)
            
        # Format conversation history
        formatted_history = self.format_conversation_history(conversation_history)
        
        # Create guiding prompt
        prompt = (
            f"You are {self.name}, {self.role}.\n"
            f"Backstory: {self.backstory}\n"
            f"Communication Style: {self.style}\n"
            f"{self.instructions}\n\n"
            f"Your task is to ensure that all participants stay on topic related to the "
            f"original question.\n"
            f"Original Question: {self.original_question or 'Unknown'}\n\n"
            f"Conversation History:\n{formatted_history}\n\n"
            f"Assess if the conversation is staying on topic. If it is, provide a brief encouraging "
            f"comment. If it's veering off-topic, provide a gentle, professional reminder to "
            f"refocus on the original question. Keep your response concise and helpful."
        )
        
        # Generate guiding response
        response_content = self.generate_response(prompt, temperature=temperature, top_p=top_p)
        
        # Update intervention state
        current_time = len(conversation_history)  # Use conversation length as proxy for time
        if self.should_intervene(formatted_history[-500:] if formatted_history else "", current_time):
            self.last_intervention_time = current_time
            self.intervention_count += 1
        
        # Save to conversation history if DB manager is available
        if self.db_manager:
            self.db_manager.save_conversation(self.agent_id, self.name, response_content)
            
        # Broadcast to global workspace if available
        if self.global_workspace:
            self.broadcast_to_global_workspace(
                content={
                    "guidance": response_content,
                    "original_question": self.original_question,
                    "is_intervention": self.should_intervene(formatted_history[-500:] if formatted_history else "", current_time)
                },
                activation=0.9,  # High priority for guidance
                metadata={
                    "type": "guidance",
                    "intervention_count": self.intervention_count
                }
            )
            
        return response_content
        
    def reset_state(self) -> None:
        """Reset the guiding agent's state for a new conversation."""
        self.original_question = None
        self.topic_keywords = []
        self.last_intervention_time = 0
        self.intervention_count = 0
        
        logger.info(f"GuidingAgent '{self.name}' state reset")

"""
Database Manager for Legion AGI

Provides database connectivity, persistence, and retrieval functions for the Legion AGI system.
Handles storing agent information, conversation history, and system state.
"""

import sqlite3
import json
import os
import datetime
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger


class DatabaseManager:
    """
    Database Manager for Legion AGI system.
    Handles database operations for persisting system state and conversation history.
    """
    
    def __init__(self, session_id: str, db_path: Optional[str] = None):
        """
        Initialize Database Manager.
        
        Args:
            session_id: Session identifier
            db_path: Path to SQLite database file
        """
        self.session_id = session_id
        self.db_path = db_path or f"session_{session_id}.db"
        
        # Ensure directory exists for the database
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.create_tables()
            logger.info(f"Database initialized at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database connection failed: {e}")
            raise
            
    def create_tables(self) -> None:
        """Create necessary database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Table for agent information
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                session_id TEXT,
                name TEXT,
                role TEXT,
                backstory TEXT,
                style TEXT,
                model TEXT,
                creation_time DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table for agent cognitive parameters
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_parameters (
                agent_id TEXT,
                parameter_name TEXT,
                parameter_value REAL,
                update_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (agent_id, parameter_name),
                FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
            )
        ''')
        
        # Table for conversation history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_history (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                agent_id TEXT,
                role TEXT,
                content TEXT,
                method TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')
        
        # Table for memory items
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_items (
                memory_id TEXT PRIMARY KEY,
                agent_id TEXT,
                content TEXT,
                memory_type TEXT,
                importance REAL,
                creation_time DATETIME,
                last_access_time DATETIME,
                access_count INTEGER,
                metadata TEXT,
                FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
            )
        ''')
        
        # Table for reasoning steps
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reasoning_steps (
                step_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                agent_id TEXT,
                goal TEXT,
                iteration INTEGER,
                method TEXT,
                reasoning_output TEXT,
                timestamp DATETIME,
                FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
            )
        ''')
        
        # Table for evolution history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evolution_history (
                evolution_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                generation INTEGER,
                population_size INTEGER,
                average_fitness REAL,
                max_fitness REAL,
                timestamp DATETIME,
                parameters TEXT
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_session ON conversation_history(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_agent ON memory_items(agent_id)")
        
        self.conn.commit()
        
    def save_agent(self, agent: Any) -> None:
        """
        Save agent information to database.
        
        Args:
            agent: Agent object to save
        """
        cursor = self.conn.cursor()
        try:
            # Save basic agent information
            cursor.execute('''
                INSERT OR REPLACE INTO agents 
                (agent_id, session_id, name, role, backstory, style, model, creation_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                agent.agent_id,
                self.session_id,
                agent.name,
                agent.role,
                agent.backstory,
                agent.style,
                agent.model,
                datetime.datetime.now().isoformat()
            ))
            
            # Save cognitive parameters
            for param_name in ['creativity', 'attention_span', 'learning_rate']:
                if hasattr(agent, param_name):
                    cursor.execute('''
                        INSERT OR REPLACE INTO agent_parameters
                        (agent_id, parameter_name, parameter_value, update_time)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        agent.agent_id,
                        param_name,
                        getattr(agent, param_name),
                        datetime.datetime.now().isoformat()
                    ))
                    
            self.conn.commit()
            logger.debug(f"Saved agent {agent.name} to database")
            
        except sqlite3.Error as e:
            logger.error(f"Error saving agent to database: {e}")
            self.conn.rollback()
            
    def save_conversation(self, agent_id: Optional[str], role: str, content: str, 
                        method: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Save conversation message to database.
        
        Args:
            agent_id: ID of the agent (None for user or system)
            role: Role of the message sender
            content: Message content
            method: Reasoning method used (if applicable)
            metadata: Additional metadata
            
        Returns:
            ID of the saved message
        """
        cursor = self.conn.cursor()
        try:
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor.execute('''
                INSERT INTO conversation_history
                (session_id, agent_id, role, content, method, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.session_id,
                agent_id,
                role,
                content,
                method,
                datetime.datetime.now().isoformat(),
                metadata_json
            ))
            
            self.conn.commit()
            message_id = cursor.lastrowid
            logger.debug(f"Saved message from {role} to conversation history")
            return message_id
            
        except sqlite3.Error as e:
            logger.error(f"Error saving conversation: {e}")
            self.conn.rollback()
            return -1
            
    def load_conversation(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load conversation history from database.
        
        Args:
            limit: Maximum number of messages to load (None for all)
            
        Returns:
            List of conversation messages
        """
        cursor = self.conn.cursor()
        try:
            if limit:
                cursor.execute('''
                    SELECT message_id, agent_id, role, content, method, timestamp, metadata
                    FROM conversation_history
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                    LIMIT ?
                ''', (self.session_id, limit))
            else:
                cursor.execute('''
                    SELECT message_id, agent_id, role, content, method, timestamp, metadata
                    FROM conversation_history
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                ''', (self.session_id,))
                
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            messages = []
            for row in rows:
                message_id, agent_id, role, content, method, timestamp, metadata_json = row
                metadata = json.loads(metadata_json) if metadata_json else None
                
                messages.append({
                    'message_id': message_id,
                    'agent_id': agent_id,
                    'role': role,
                    'content': content,
                    'method': method,
                    'timestamp': timestamp,
                    'metadata': metadata
                })
                
            return messages
            
        except sqlite3.Error as e:
            logger.error(f"Error loading conversation: {e}")
            return []
            
    def save_memory_item(self, agent_id: str, memory_item: Dict[str, Any]) -> None:
        """
        Save memory item to database.
        
        Args:
            agent_id: ID of the agent
            memory_item: Memory item dictionary
        """
        cursor = self.conn.cursor()
        try:
            memory_id = memory_item.get('id', 'unknown')
            content = memory_item.get('content', '')
            memory_type = memory_item.get('memory_type', 'unknown')
            importance = memory_item.get('importance', 0.5)
            creation_time = memory_item.get('timestamp', datetime.datetime.now().isoformat())
            last_access_time = memory_item.get('last_accessed', creation_time)
            access_count = memory_item.get('access_count', 0)
            metadata = json.dumps(memory_item.get('metadata', {}))
            
            cursor.execute('''
                INSERT OR REPLACE INTO memory_items
                (memory_id, agent_id, content, memory_type, importance, 
                creation_time, last_access_time, access_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                memory_id,
                agent_id,
                content,
                memory_type,
                importance,
                creation_time,
                last_access_time,
                access_count,
                metadata
            ))
            
            self.conn.commit()
            logger.debug(f"Saved memory item {memory_id} for agent {agent_id}")
            
        except sqlite3.Error as e:
            logger.error(f"Error saving memory item: {e}")
            self.conn.rollback()
            
    def load_agent_memories(self, agent_id: str, memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load memory items for an agent.
        
        Args:
            agent_id: ID of the agent
            memory_type: Optional type filter
            
        Returns:
            List of memory items
        """
        cursor = self.conn.cursor()
        try:
            if memory_type:
                cursor.execute('''
                    SELECT memory_id, content, memory_type, importance, creation_time, 
                    last_access_time, access_count, metadata
                    FROM memory_items
                    WHERE agent_id = ? AND memory_type = ?
                    ORDER BY importance DESC
                ''', (agent_id, memory_type))
            else:
                cursor.execute('''
                    SELECT memory_id, content, memory_type, importance, creation_time, 
                    last_access_time, access_count, metadata
                    FROM memory_items
                    WHERE agent_id = ?
                    ORDER BY importance DESC
                ''', (agent_id,))
                
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            memories = []
            for row in rows:
                memory_id, content, mem_type, importance, creation_time, last_access, access_count, metadata_json = row
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                memories.append({
                    'id': memory_id,
                    'content': content,
                    'memory_type': mem_type,
                    'importance': importance,
                    'timestamp': creation_time,
                    'last_accessed': last_access,
                    'access_count': access_count,
                    'metadata': metadata
                })
                
            return memories
            
        except sqlite3.Error as e:
            logger.error(f"Error loading agent memories: {e}")
            return []
            
    def save_reasoning_step(self, session_id: str, agent_id: str, reasoning_step: Dict[str, Any]) -> None:
        """
        Save reasoning step to database.
        
        Args:
            session_id: Session ID
            agent_id: Agent ID
            reasoning_step: Reasoning step dictionary
        """
        cursor = self.conn.cursor()
        try:
            goal = reasoning_step.get('goal', '')
            iteration = reasoning_step.get('iteration', 0)
            method = reasoning_step.get('method', 'unknown')
            reasoning_output = reasoning_step.get('reasoning_output', '')
            timestamp = reasoning_step.get('timestamp', datetime.datetime.now().isoformat())
            
            cursor.execute('''
                INSERT INTO reasoning_steps
                (session_id, agent_id, goal, iteration, method, reasoning_output, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                agent_id,
                goal,
                iteration,
                method,
                reasoning_output,
                timestamp
            ))
            
            self.conn.commit()
            logger.debug(f"Saved reasoning step for agent {agent_id}")
            
        except sqlite3.Error as e:
            logger.error(f"Error saving reasoning step: {e}")
            self.conn.rollback()
            
    def load_reasoning_chain(self, agent_id: str, goal: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load reasoning chain for an agent.
        
        Args:
            agent_id: Agent ID
            goal: Optional goal filter
            
        Returns:
            List of reasoning steps
        """
        cursor = self.conn.cursor()
        try:
            if goal:
                cursor.execute('''
                    SELECT step_id, goal, iteration, method, reasoning_output, timestamp
                    FROM reasoning_steps
                    WHERE agent_id = ? AND goal = ?
                    ORDER BY iteration ASC
                ''', (agent_id, goal))
            else:
                cursor.execute('''
                    SELECT step_id, goal, iteration, method, reasoning_output, timestamp
                    FROM reasoning_steps
                    WHERE agent_id = ?
                    ORDER BY timestamp DESC, iteration ASC
                ''', (agent_id,))
                
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            steps = []
            for row in rows:
                step_id, goal, iteration, method, reasoning_output, timestamp = row
                
                steps.append({
                    'step_id': step_id,
                    'goal': goal,
                    'iteration': iteration,
                    'method': method,
                    'reasoning_output': reasoning_output,
                    'timestamp': timestamp
                })
                
            return steps
            
        except sqlite3.Error as e:
            logger.error(f"Error loading reasoning chain: {e}")
            return []
            
    def save_evolution_generation(self, generation_data: Dict[str, Any]) -> None:
        """
        Save evolution generation data to database.
        
        Args:
            generation_data: Generation data dictionary
        """
        cursor = self.conn.cursor()
        try:
            generation = generation_data.get('generation', 0)
            population_size = generation_data.get('population_size', 0)
            average_fitness = generation_data.get('average_fitness', 0.0)
            max_fitness = generation_data.get('max_fitness', 0.0)
            timestamp = generation_data.get('timestamp', datetime.datetime.now().isoformat())
            parameters = json.dumps(generation_data.get('parameters', {}))
            
            cursor.execute('''
                INSERT INTO evolution_history
                (session_id, generation, population_size, average_fitness, max_fitness, timestamp, parameters)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.session_id,
                generation,
                population_size,
                average_fitness,
                max_fitness,
                timestamp,
                parameters
            ))
            
            self.conn.commit()
            logger.debug(f"Saved evolution generation {generation} data")
            
        except sqlite3.Error as e:
            logger.error(f"Error saving evolution data: {e}")
            self.conn.rollback()
            
    def load_evolution_history(self) -> List[Dict[str, Any]]:
        """
        Load evolution history from database.
        
        Returns:
            List of evolution generation data
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                SELECT evolution_id, generation, population_size, average_fitness, max_fitness, timestamp, parameters
                FROM evolution_history
                WHERE session_id = ?
                ORDER BY generation ASC
            ''', (self.session_id,))
            
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            history = []
            for row in rows:
                evolution_id, generation, population_size, average_fitness, max_fitness, timestamp, parameters_json = row
                parameters = json.loads(parameters_json) if parameters_json else {}
                
                history.append({
                    'evolution_id': evolution_id,
                    'generation': generation,
                    'population_size': population_size,
                    'average_fitness': average_fitness,
                    'max_fitness': max_fitness,
                    'timestamp': timestamp,
                    'parameters': parameters
                })
                
            return history
            
        except sqlite3.Error as e:
            logger.error(f"Error loading evolution history: {e}")
            return []
            
    def get_agent_by_id(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get agent information by ID.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent information dictionary or None if not found
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                SELECT agent_id, name, role, backstory, style, model, creation_time
                FROM agents
                WHERE agent_id = ?
            ''', (agent_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
                
            agent_id, name, role, backstory, style, model, creation_time = row
                
            # Get cognitive parameters
            cursor.execute('''
                SELECT parameter_name, parameter_value
                FROM agent_parameters
                WHERE agent_id = ?
            ''', (agent_id,))
            
            param_rows = cursor.fetchall()
            parameters = {name: value for name, value in param_rows}
                
            return {
                'agent_id': agent_id,
                'name': name,
                'role': role,
                'backstory': backstory,
                'style': style,
                'model': model,
                'creation_time': creation_time,
                'parameters': parameters
            }
                
        except sqlite3.Error as e:
            logger.error(f"Error getting agent by ID: {e}")
            return None
            
    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.debug("Database connection closed")

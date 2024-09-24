# Import necessary libraries for Congition: LegionAI
import ollama
import random
import sqlite3
import uuid
import datetime
import os
import json
import re
from typing import List, Dict, Any, Optional
from loguru import logger

# Configure Logging to Write to 'debug_log.txt'
logger.add("debug_log.txt", rotation="10 MB", level="DEBUG")

# Database Manager Class
class DatabaseManager:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.db_name = f"session_{session_id}.db"
        try:
            self.conn = sqlite3.connect(self.db_name)
            self.create_tables()
        except sqlite3.Error as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def create_tables(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                name TEXT,
                role TEXT,
                backstory TEXT,
                style TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                agent_id TEXT,
                role TEXT,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def save_agent(self, agent) -> None:
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO agents (agent_id, name, role, backstory, style)
                VALUES (?, ?, ?, ?, ?)
            ''', (agent.agent_id, agent.name, agent.role, agent.backstory, agent.style))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error saving agent to database: {e}")

    def save_conversation(self, agent_id: Optional[str], role: str, content: str) -> None:
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO conversation_history (session_id, agent_id, role, content)
                VALUES (?, ?, ?, ?)
            ''', (self.session_id, agent_id, role, content))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error saving conversation: {e}")

    def load_conversation(self) -> List[tuple]:
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                SELECT agent_id, role, content FROM conversation_history
                WHERE session_id = ?
                ORDER BY timestamp ASC
            ''', (self.session_id,))
            return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Error loading conversation: {e}")
            return []

    def close(self) -> None:
        self.conn.close()

# Base Agent Class with Collaborative Reasoning
class Agent:
    def __init__(self, name: str, role: str, backstory: str, style: str, instructions: str, model: str = "llama3.1:8b", db_manager: DatabaseManager = None):
        self.agent_id = str(uuid.uuid4())
        self.name = name
        self.role = role
        self.backstory = backstory
        self.style = style
        self.instructions = instructions
        self.model = model
        self.agent_list: List['Agent'] = []
        self.prompt: str = ""
        self.db_manager = db_manager

        if self.db_manager:
            self.db_manager.save_agent(self)

    def __repr__(self) -> str:
        return f"Agent(name={self.name}, role={self.role})"

    def set_agent_list(self, agent_list: List['Agent']) -> None:
        self.agent_list = agent_list
        self.prompt = self.create_prompt()

    def create_prompt(self) -> str:
        agent_names = ', '.join([agent.name for agent in self.agent_list if agent.name != self.name])
        persona = (
            f"You are {self.name}, {self.role}.\n"
            f"Backstory: {self.backstory}\n"
            f"Communication Style: {self.style}\n"
            f"{self.instructions}\n"
            f"Participants in the conversation: {agent_names}.\n"
            "Your task is to collaboratively brainstorm and build upon the previous discussions to contribute to a comprehensive solution.\n"
            "Respond in first person singular.\n"
            "Do not mention that you are an AI language model.\n"
            "Stay focused on the original how-to question and avoid going off-topic.\n"
        )
        return persona

    def respond(self, conversation_history: List[Dict[str, str]], temperature: float = 0.7, top_p: float = 0.9) -> str:
        formatted_history = self.format_conversation_history(conversation_history)
        prompt = self.prompt + "\nConversation History:\n" + formatted_history + f"{self.name}:"
        messages = [{'role': 'user', 'content': prompt}]
        try:
            logger.debug(f"{self.name} is generating a response with prompt:\n{prompt}")
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={'temperature': temperature, 'top_p': top_p, 'max_tokens': 500}
            )
            # Access the response content correctly
            response_content = response['message']['content'].strip()
            logger.debug(f"{self.name}'s response:\n{response_content}")
            if self.db_manager:
                self.db_manager.save_conversation(self.agent_id, self.name, response_content)
            return response_content
        except Exception as e:
            logger.error(f"Error generating response for {self.name}: {e}")
            return f"Error generating response: {e}"

    def format_conversation_history(self, conversation_history: List[Dict[str, str]], limit: Optional[int] = None) -> str:
        formatted_history = ""
        for entry in conversation_history[-limit:] if limit else conversation_history:
            role = entry['role']
            content = entry['content']
            formatted_history += f"{role}: {content}\n"
        return formatted_history

# Guiding Agent Class to Keep Conversations On Topic
class GuidingAgent(Agent):
    def __init__(self, name: str, role: str, backstory: str, style: str, instructions: str, model: str = "llama3.1:8b", db_manager: DatabaseManager = None):
        super().__init__(name, role, backstory, style, instructions, model, db_manager)

    def respond(self, conversation_history: List[Dict[str, str]], original_question: str, temperature: float = 0.5, top_p: float = 0.7) -> str:
        # Analyze the conversation and provide reminders if off-topic
        formatted_history = self.format_conversation_history(conversation_history)
        prompt = (
            f"You are {self.name}, {self.role}.\n"
            f"Backstory: {self.backstory}\n"
            f"Communication Style: {self.style}\n"
            f"{self.instructions}\n"
            "Your task is to ensure that all participants stay on topic related to the original how-to question.\n"
            "If you detect that the conversation is veering off-topic, politely remind the participants to focus on the main question.\n"
            f"Original Question: {original_question}\n\n"
            f"Conversation History:\n{formatted_history}\n\n"
            "Provide a gentle reminder to keep the discussion on track if necessary."
        )
        messages = [{'role': 'user', 'content': prompt}]
        try:
            logger.debug(f"{self.name} is analyzing the conversation for focus.")
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={'temperature': temperature, 'top_p': top_p, 'max_tokens': 150}
            )
            response_content = response['message']['content'].strip()
            logger.debug(f"{self.name}'s reminder:\n{response_content}")
            if self.db_manager:
                self.db_manager.save_conversation(self.agent_id, self.name, response_content)
            return response_content
        except Exception as e:
            logger.error(f"Error generating response for {self.name}: {e}")
            return f"Error generating response: {e}"

# State Evaluator Agent
class StateEvaluator:
    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model

    def evaluate(self, reasoning_chain: str) -> float:
        prompt = (
            "Evaluate the following reasoning for logic, coherence, and relevance to the problem.\n"
            f"Reasoning:\n{reasoning_chain}\n"
            "Score the reasoning on a scale from 0 to 1, where 1 is the highest score.\n"
            "Please provide the score first, followed by any additional comments.\n"
            "Score:"
        )
        messages = [{'role': 'user', 'content': prompt}]
        try:
            logger.debug(f"Evaluating reasoning chain:\n{reasoning_chain}")
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={'temperature': 0.0, 'top_p': 0.0}
            )
            response_content = response['message']['content'].strip()
            logger.debug(f"LLM Response: {response_content}")
            match = re.search(r"([0-1](?:\.\d+)?)", response_content)
            if match:
                score_str = match.group(1)
                score = float(score_str)
                logger.debug(f"Assigned score: {score}")
                return score
            else:
                logger.error("Could not find a numerical score in the LLM's response.")
                return 0.0
        except Exception as e:
            logger.error(f"Error evaluating reasoning chain: {e}")
            return 0.0

# Thought Validator Agent
class ThoughtValidator:
    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model

    def validate(self, reasoning_chain: str) -> bool:
        prompt = (
            "As a Thought Validator, assess the following reasoning chain for logical consistency, "
            "factual accuracy, and completeness. Respond with 'Validated' if it passes all checks "
            "or 'Invalidated' if it fails any.\n"
            f"Reasoning Chain:\n{reasoning_chain}\n"
            "Validation Result:"
        )
        messages = [{'role': 'user', 'content': prompt}]
        try:
            logger.debug(f"Validating reasoning chain:\n{reasoning_chain}")
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={'temperature': 0.0, 'top_p': 0.0}
            )
            result = response['message']['content'].strip()
            logger.debug(f"Validation result: {result}")
            return 'Validated' in result
        except Exception as e:
            logger.error(f"Error validating reasoning chain: {e}")
            return False

# Refinement Agent for RAFT Phase
class RefinementAgent(Agent):
    def __init__(self, name: str, role: str, backstory: str, style: str, instructions: str, model: str = "llama3.1:8b", db_manager: DatabaseManager = None):
        super().__init__(name, role, backstory, style, instructions, model, db_manager)

    def respond(self, conversation_history: List[Dict[str, str]], original_question: str, temperature: float = 0.6, top_p: float = 0.8) -> str:
        # Provide refinement suggestions based on previous solutions
        formatted_history = self.format_conversation_history(conversation_history)
        prompt = (
            f"You are {self.name}, {self.role}.\n"
            f"Backstory: {self.backstory}\n"
            f"Communication Style: {self.style}\n"
            f"{self.instructions}\n"
            f"Participants in the conversation: {', '.join([agent.name for agent in self.agent_list if agent.name != self.name])}.\n"
            "Your task is to refine the existing solutions by providing feedback, identifying potential improvements, and suggesting actionable enhancements.\n"
            "Respond in first person singular.\n"
            "Do not mention that you are an AI language model.\n"
            "Stay focused on refining the solutions related to the original how-to question.\n\n"
            f"Original Question: {original_question}\n\n"
            f"Conversation History:\n{formatted_history}\n\n"
            f"{self.name}:"
        )
        messages = [{'role': 'user', 'content': prompt}]
        try:
            logger.debug(f"{self.name} is generating refinement suggestions with prompt:\n{prompt}")
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={'temperature': temperature, 'top_p': top_p, 'max_tokens': 500}
            )
            response_content = response['message']['content'].strip()
            logger.debug(f"{self.name}'s refinement suggestions:\n{response_content}")
            if self.db_manager:
                self.db_manager.save_conversation(self.agent_id, self.name, response_content)
            return response_content
        except Exception as e:
            logger.error(f"Error generating response for {self.name}: {e}")
            return f"Error generating response: {e}"

# Evaluation Agent for EAT Phase
class EvaluationAgent(Agent):
    def __init__(self, name: str, role: str, backstory: str, style: str, instructions: str, model: str = "llama3.1:8b", db_manager: DatabaseManager = None):
        super().__init__(name, role, backstory, style, instructions, model, db_manager)

    def respond(self, conversation_history: List[Dict[str, str]], original_question: str, temperature: float = 0.6, top_p: float = 0.8) -> str:
        # Evaluate the refined solutions for feasibility and effectiveness
        formatted_history = self.format_conversation_history(conversation_history)
        prompt = (
            f"You are {self.name}, {self.role}.\n"
            f"Backstory: {self.backstory}\n"
            f"Communication Style: {self.style}\n"
            f"{self.instructions}\n"
            f"Participants in the conversation: {', '.join([agent.name for agent in self.agent_list if agent.name != self.name])}.\n"
            "Your task is to evaluate the refined solutions for their feasibility, effectiveness, and practicality.\n"
            "Provide actionable feedback and suggest any necessary modifications to ensure successful implementation.\n"
            "Respond in first person singular.\n"
            "Do not mention that you are an AI language model.\n"
            "Stay focused on evaluating solutions related to the original how-to question.\n\n"
            f"Original Question: {original_question}\n\n"
            f"Conversation History:\n{formatted_history}\n\n"
            f"{self.name}:"
        )
        messages = [{'role': 'user', 'content': prompt}]
        try:
            logger.debug(f"{self.name} is evaluating solutions with prompt:\n{prompt}")
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={'temperature': temperature, 'top_p': top_p, 'max_tokens': 500}
            )
            response_content = response['message']['content'].strip()
            logger.debug(f"{self.name}'s evaluation feedback:\n{response_content}")
            if self.db_manager:
                self.db_manager.save_conversation(self.agent_id, self.name, response_content)
            return response_content
        except Exception as e:
            logger.error(f"Error generating response for {self.name}: {e}")
            return f"Error generating response: {e}"

# Final Agent to Compile the How-To Manual
class FinalAgent:
    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model

    def compile_manual(self, topic: str, contributions: List[str], additional_text: str) -> str:
        prompt = (
            f"Using the following contributions from experts and the additional text provided, write a comprehensive how-to manual on the topic.\n"
            f"Topic: {topic}\n"
            "Contributions:\n"
            + "\n\n".join(contributions) +
            "\n\nAdditional Text:\n" + additional_text +
            "\n\n"
            "Write the how-to manual in a clear and concise style with proper structure, including Introduction, Materials Needed, Step-by-Step Instructions, Tips and Tricks, and Conclusion.\n"
            "Ensure logical coherence and incorporate the key points from the contributions and the additional text.\n"
            "Do not mention the agents or the conversation.\n"
            "Focus on practical applications and actionable steps.\n"
            "Begin your how-to manual now."
        )
        messages = [{'role': 'user', 'content': prompt}]
        try:
            logger.debug(f"FinalAgent is compiling the how-to manual with prompt:\n{prompt}")
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={'temperature': 0.7, 'top_p': 0.9, 'max_tokens': 2000}
            )
            manual = response['message']['content'].strip()
            logger.debug(f"Final how-to manual:\n{manual}")
            return manual
        except Exception as e:
            logger.error(f"Error compiling how-to manual: {e}")
            return "Error compiling how-to manual."

# Chat Manager Class with Collaborative Reasoning
class ChatManager:
    def __init__(self):
        self.agents: List[Agent] = []
        self.guiding_agent: Optional[GuidingAgent] = None
        self.refinement_agent: Optional[RefinementAgent] = None
        self.evaluation_agent: Optional[EvaluationAgent] = None
        self.conversation_history: List[Dict[str, str]] = []
        self.db_manager = DatabaseManager(session_id=str(uuid.uuid4()))
        self.state_evaluator = StateEvaluator()
        self.thought_validator = ThoughtValidator()
        self.final_agent = FinalAgent()
        self.iteration_count = 0
        self.max_iterations = 10  # Increased for deeper iterations
        self.original_question: Optional[str] = None
        self.additional_text: str = ""
        self.phase = "PAST"  # Initial Phase

    def analyze_input_and_spawn_agents(self, message: str) -> None:
        prompt = (
            "Analyze the following user input and suggest as many experts (up to 10, real non-fictional people) "
            "that could help solve the problem. Provide their names, roles, backstories, communication styles, "
            "and specific instructions for collaboration in the following JSON format:\n\n"
            "[\n"
            "  {\n"
            "    \"Name\": \"Expert's Name\",\n"
            "    \"Role\": \"Expert's Role\",\n"
            "    \"Backstory\": \"Expert's Backstory\",\n"
            "    \"Style\": \"Expert's Communication Style\",\n"
            "    \"Instructions\": \"Specific instructions for the agent\"\n"
            "  },\n"
            "  ...\n"
            "]\n\n"
            f"User Input: \"{message}\"\n\n"
            "**Please output only the JSON data and nothing else. Ensure that all JSON syntax is correct, including commas between fields and objects.**"
        )
        messages = [{'role': 'user', 'content': prompt}]
        try:
            logger.debug("Requesting LLM to suggest experts...")
            response = ollama.chat(model="llama3.1:8b", messages=messages)
            suggested_agents = response['message']['content'].strip()
            logger.debug(f"LLM Response for Agent Spawning:\n{suggested_agents}\n")
            self.parse_and_create_agents(suggested_agents)
        except Exception as e:
            logger.error(f"Error analyzing input: {e}")

    def parse_and_create_agents(self, suggested_agents_text: str) -> None:
        agents = []
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Attempt to parse JSON
                agent_list = json.loads(suggested_agents_text)
                for agent_info in agent_list:
                    agent = Agent(
                        name=agent_info.get('Name', 'Unknown'),
                        role=agent_info.get('Role', ''),
                        backstory=agent_info.get('Backstory', ''),
                        style=agent_info.get('Style', ''),
                        instructions=agent_info.get('Instructions', 'Collaborate effectively.'),
                        model="llama3.1:8b",
                        db_manager=self.db_manager
                    )
                    agents.append(agent)
                if not agents:
                    logger.error("No agents were parsed from the LLM's response.")
                else:
                    for agent in agents:
                        agent.set_agent_list(agents)
                        logger.info(f"Agent '{agent.name}' has been created.")
                    self.agents = agents

                # Add the GuidingAgent
                guiding_agent_info = {
                    "Name": "GuidingAgent",
                    "Role": "Conversation Moderator",
                    "Backstory": "You are a seasoned moderator trained to keep discussions focused and productive.",
                    "Style": "Polite and assertive.",
                    "Instructions": "Monitor the conversation and ensure all agents stay on topic."
                }
                guiding_agent = GuidingAgent(
                    name=guiding_agent_info["Name"],
                    role=guiding_agent_info["Role"],
                    backstory=guiding_agent_info["Backstory"],
                    style=guiding_agent_info["Style"],
                    instructions=guiding_agent_info["Instructions"],
                    model="llama3.1:8b",
                    db_manager=self.db_manager
                )
                self.guiding_agent = guiding_agent
                self.agents.append(guiding_agent)
                logger.info(f"GuidingAgent '{guiding_agent.name}' has been created.")
                break  # Successfully parsed and created agents
            except json.JSONDecodeError as e:
                logger.error(f"JSON decoding error: {e}")
                logger.debug(f"Attempt {retry_count + 1} of {max_retries}: LLM's response was: {suggested_agents_text}")
                retry_count += 1
                if retry_count < max_retries:
                    # Attempt to correct common JSON issues
                    suggested_agents_text = self.correct_json(suggested_agents_text)
                else:
                    logger.error("Maximum retries reached. Could not parse agents.")
                    print("Error: Unable to parse agent information. Please try again later.")
                    return
            except Exception as e:
                logger.error(f"Error parsing agents: {e}")
                logger.debug(f"LLM's response was: {suggested_agents_text}")
                print("Error: Unable to parse agent information. Please try again later.")
                return

    def correct_json(self, json_text: str) -> str:
        """
        Attempt to correct common JSON formatting issues.
        This is a basic implementation and may need to be extended for more complex errors.
        """
        # Add missing commas between objects
        json_text = re.sub(r'}\s*{', r'}, {', json_text)
        # Ensure commas between key-value pairs
        json_text = re.sub(r'"\s*"Style', r'", "Style', json_text)
        # Add commas after the last field if missing
        json_text = re.sub(r'("Instructions": ".*?")\s*}', r'\1}', json_text)
        return json_text

    def run_collaborative_reasoning(self, message: str) -> Optional[str]:
        if self.phase == "PAST":
            # Phase 1: Initial Spawning
            self.iteration_count += 1
            logger.info(f"--- Phase 1: Initial Spawning ---\n")
            combined_conversation = self.conversation_history.copy()
            contributions = []
            reasoning_branches: Dict[str, str] = {}
            validated_branches: Dict[str, str] = {}

            for agent in self.agents:
                if isinstance(agent, GuidingAgent):
                    # GuidingAgent performs its role
                    reminder = agent.respond(combined_conversation, self.original_question)
                    combined_conversation.append({'role': agent.name, 'content': reminder})
                    self.conversation_history.append({'role': agent.name, 'content': reminder})
                    logger.info(f"{agent.name} provided a reminder to stay on topic.\n")
                    continue

                logger.info(f"{agent.name} is generating response...\n")
                agent_response = agent.respond(combined_conversation)
                combined_conversation.append({'role': agent.name, 'content': agent_response})
                self.conversation_history.append({'role': agent.name, 'content': agent_response})
                contributions.append(agent_response)
                reasoning_branches[agent.name] = agent_response

                # GuidingAgent checks after each agent's response
                if self.guiding_agent:
                    reminder = self.guiding_agent.respond(combined_conversation, self.original_question)
                    if reminder:
                        combined_conversation.append({'role': self.guiding_agent.name, 'content': reminder})
                        self.conversation_history.append({'role': self.guiding_agent.name, 'content': reminder})
                        logger.info(f"{self.guiding_agent.name} provided a reminder to stay on topic.\n")

            # Thought Validator evaluates each reasoning branch
            for agent_name, reasoning_chain in reasoning_branches.items():
                is_valid = self.thought_validator.validate(reasoning_chain)
                if is_valid:
                    validated_branches[agent_name] = reasoning_chain
                    logger.info(f"{agent_name}'s reasoning is Validated.\n")
                else:
                    logger.info(f"{agent_name}'s reasoning is Invalidated.\n")

            if validated_branches:
                valid_contributions = [reasoning_chain for reasoning_chain in validated_branches.values()]
                final_manual = self.final_agent.compile_manual(self.original_question, valid_contributions, self.additional_text)
                if final_manual:
                    self.save_final_solution(final_manual)
                    self.phase = "ADVANCED_REASONING"
                    return final_manual
                else:
                    logger.info("Unable to compile final how-to manual.\n")
                    return None
            else:
                logger.info("No validated reasoning branches. Cannot produce a final how-to manual.\n")
                return None

        elif self.phase == "ADVANCED_REASONING":
            # Phase 2: Advanced Reasoning (RAFT/EAT/CIR)
            self.iteration_count += 1
            logger.info(f"--- Phase 2: Advanced Reasoning ---\n")
            combined_conversation = self.conversation_history.copy()
            contributions = []
            reasoning_branches: Dict[str, str] = {}
            validated_branches: Dict[str, str] = {}

            # Initialize Refinement and Evaluation Agents if not already
            if not self.refinement_agent:
                refinement_agent_info = {
                    "Name": "RefinementAgent",
                    "Role": "Refinement Specialist",
                    "Backstory": "You are an expert in refining and improving solutions through structured feedback.",
                    "Style": "Constructive and analytical.",
                    "Instructions": "Provide feedback and suggest improvements to the existing solutions."
                }
                self.refinement_agent = RefinementAgent(
                    name=refinement_agent_info["Name"],
                    role=refinement_agent_info["Role"],
                    backstory=refinement_agent_info["Backstory"],
                    style=refinement_agent_info["Style"],
                    instructions=refinement_agent_info["Instructions"],
                    model="llama3.1:8b",
                    db_manager=self.db_manager
                )
                self.agents.append(self.refinement_agent)
                logger.info(f"RefinementAgent '{self.refinement_agent.name}' has been created.")

            if not self.evaluation_agent:
                evaluation_agent_info = {
                    "Name": "EvaluationAgent",
                    "Role": "Solution Evaluator",
                    "Backstory": "You are an expert in evaluating the feasibility and effectiveness of solutions.",
                    "Style": "Critical and thorough.",
                    "Instructions": "Assess the solutions for their practicality and suggest necessary modifications."
                }
                self.evaluation_agent = EvaluationAgent(
                    name=evaluation_agent_info["Name"],
                    role=evaluation_agent_info["Role"],
                    backstory=evaluation_agent_info["Backstory"],
                    style=evaluation_agent_info["Style"],
                    instructions=evaluation_agent_info["Instructions"],
                    model="llama3.1:8b",
                    db_manager=self.db_manager
                )
                self.agents.append(self.evaluation_agent)
                logger.info(f"EvaluationAgent '{self.evaluation_agent.name}' has been created.")

            for agent in self.agents:
                if isinstance(agent, (GuidingAgent, RefinementAgent, EvaluationAgent)):
                    # These agents perform their specialized roles
                    if isinstance(agent, RefinementAgent):
                        refinement_response = agent.respond(combined_conversation, self.original_question)
                        combined_conversation.append({'role': agent.name, 'content': refinement_response})
                        self.conversation_history.append({'role': agent.name, 'content': refinement_response})
                        contributions.append(refinement_response)
                        reasoning_branches[agent.name] = refinement_response
                        logger.info(f"{agent.name} provided refinement suggestions.\n")
                    elif isinstance(agent, EvaluationAgent):
                        evaluation_response = agent.respond(combined_conversation, self.original_question)
                        combined_conversation.append({'role': agent.name, 'content': evaluation_response})
                        self.conversation_history.append({'role': agent.name, 'content': evaluation_response})
                        contributions.append(evaluation_response)
                        reasoning_branches[agent.name] = evaluation_response
                        logger.info(f"{agent.name} provided evaluation feedback.\n")
                    else:
                        # GuidingAgent performs its role
                        reminder = agent.respond(combined_conversation, self.original_question)
                        combined_conversation.append({'role': agent.name, 'content': reminder})
                        self.conversation_history.append({'role': agent.name, 'content': reminder})
                        logger.info(f"{agent.name} provided a reminder to stay on topic.\n")
                    continue

                logger.info(f"{agent.name} is refining their response...\n")
                agent_response = agent.respond(combined_conversation)
                combined_conversation.append({'role': agent.name, 'content': agent_response})
                self.conversation_history.append({'role': agent.name, 'content': agent_response})
                contributions.append(agent_response)
                reasoning_branches[agent.name] = agent_response

                # GuidingAgent checks after each agent's response
                if self.guiding_agent:
                    reminder = self.guiding_agent.respond(combined_conversation, self.original_question)
                    if reminder:
                        combined_conversation.append({'role': self.guiding_agent.name, 'content': reminder})
                        self.conversation_history.append({'role': self.guiding_agent.name, 'content': reminder})
                        logger.info(f"{self.guiding_agent.name} provided a reminder to stay on topic.\n")

            # Thought Validator evaluates each reasoning branch
            for agent_name, reasoning_chain in reasoning_branches.items():
                is_valid = self.thought_validator.validate(reasoning_chain)
                if is_valid:
                    validated_branches[agent_name] = reasoning_chain
                    logger.info(f"{agent_name}'s reasoning is Validated.\n")
                else:
                    logger.info(f"{agent_name}'s reasoning is Invalidated.\n")

            if validated_branches:
                valid_contributions = [reasoning_chain for reasoning_chain in validated_branches.values()]
                final_manual = self.final_agent.compile_manual(self.original_question, valid_contributions, self.additional_text)
                if final_manual:
                    self.save_final_solution(final_manual)
                    return final_manual
                else:
                    logger.info("Unable to compile final how-to manual.\n")
                    return None
            else:
                logger.info("No validated reasoning branches. Cannot produce a final how-to manual.\n")
                return None

    def save_final_solution(self, solution: str) -> None:
        if self.original_question:
            filename = f"HOWTO.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(solution)
            logger.info(f"Final how-to manual saved to {filename}")
        else:
            logger.error("Original question not found. Cannot save final solution.")

    def close_session(self) -> None:
        self.db_manager.close()

    def visualize_thought_process(self, step: str) -> None:
        # Simple visualization via logging; can be enhanced as needed
        print(f"Visualization: {step}")
        logger.info(f"Visualization: {step}")

    def start_chat(self) -> None:
        print("Welcome to the Advanced Multi-Agent Chat Session!")
        print("Type '/exit' to quit the program.\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                print("\nEnding the session. Goodbye!")
                self.close_session()
                break

            if user_input.lower() in ['/exit', '/quit']:
                print("Ending the session. Goodbye!")
                self.close_session()
                break
            elif user_input == '':
                continue
            else:
                if not self.original_question:
                    self.original_question = user_input
                    self.visualize_thought_process("Step 1: Analyzing the situation...")
                    self.analyze_input_and_spawn_agents(user_input)
                    if not self.agents:
                        print("No agents were spawned. Unable to proceed.")
                        continue
                    self.visualize_thought_process("Step 2: Determining the task...")
                else:
                    # User provided additional input
                    self.additional_text += f"\n{user_input}"
                    self.visualize_thought_process("Step 3: Agents are brainstorming solutions...")

                print("Please provide any additional text or references you would like to include (or press Enter to skip):")
                additional_input = input().strip()
                if additional_input:
                    self.additional_text += f"\n{additional_input}"

                self.conversation_history.append({'role': 'User', 'content': user_input})
                self.db_manager.save_conversation(None, 'User', user_input)

                # Visualization Step
                self.visualize_thought_process("Step 4: Agents are refining their thoughts...")

                # Run collaborative reasoning based on current phase
                final_manual = self.run_collaborative_reasoning(user_input)

                if final_manual:
                    print(f"Final How-To Manual:\n{final_manual}\n")
                else:
                    print("Unable to produce a final how-to manual.\n")
                # Continue the loop for continuous interaction

# Main Function
def main():
    chat_manager = ChatManager()
    chat_manager.start_chat()

if __name__ == "__main__":
    main()

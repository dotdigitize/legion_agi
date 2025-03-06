### mail.lite.py is a light weight one file version of LegionAGI based on the Alpha development focusing on spawning agents only###
import ollama
import random
import sqlite3
import uuid
import datetime
import os
import json
import re
from typing import List, Dict, Any, Optional
from loguru import logger as app_logger  # Rename to avoid conflict with Brian2

# Additional Libraries for Quantum-Inspired Computing and SNN
import numpy as np
import scipy.linalg
import scipy.sparse
import sympy as sp
import qiskit
import pennylane as qml
from pennylane import numpy as pnp
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import brian2 as b2
from brian2 import input as brian_input
from brian2 import ms, NeuronGroup, SpikeMonitor, start_scope, run  # Import ms and other components

# Configure Logging to Write to 'debug_log.txt'
app_logger.add("debug_log.txt", rotation="10 MB", level="DEBUG")

# Quantum-Inspired Memory Mechanism with von Neumann Operations Algebra

class QuantumMemory:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.dev = qml.device('default.qubit', wires=num_qubits)
        self.state = None  # Stores the density matrix

    def initialize_state(self, state_vector):
        # Initialize with a pure state represented by a density matrix
        ket = np.array(state_vector, dtype=complex)
        bra = ket.conj().T
        self.state = np.outer(ket, bra)

    def apply_operator(self, operator):
        # Apply a quantum operator (superoperator in Liouville space)
        # operator is a function that takes and returns a density matrix
        self.state = operator(self.state)

    def von_neumann_measurement(self, observable):
        # Perform a von Neumann measurement with respect to the given observable
        # Diagonalize the observable
        eigenvalues, eigenvectors = np.linalg.eigh(observable)
        probabilities = np.array([np.trace(np.dot(eigvec[:, np.newaxis], eigvec[np.newaxis, :].conj()).T @ self.state)
                                  for eigvec in eigenvectors.T])
        probabilities = np.real(probabilities)
        probabilities /= np.sum(probabilities)  # Normalize probabilities

        # Choose a measurement outcome based on probabilities
        outcome = np.random.choice(len(eigenvalues), p=probabilities)
        measured_eigenstate = np.dot(eigenvectors[:, outcome], eigenvectors[:, outcome].conj().T)
        measured_eigenstate = measured_eigenstate[np.newaxis, :, :].T  # Reshape for consistency
        self.state = measured_eigenstate
        return eigenvalues[outcome]

    def measure(self, observable=None):
        # Perform a measurement using the von Neumann measurement method
        if observable is None:
            # Default to computational basis measurement
            observable = np.eye(2**self.num_qubits)
        return self.von_neumann_measurement(observable)

# Spiking Neural Network Memory Module using Brian2
class SpikingNeuron:
    def __init__(self):
        start_scope()
        self.neuron = NeuronGroup(
            1,
            '''
            dv/dt = (I - v) / (10*ms) : 1
            I : 1
            ''',
            threshold='v > 1',
            reset='v = 0',
            method='euler'
        )
        self.spike_monitor = SpikeMonitor(self.neuron)
        self.neuron.I = 0  # Initialize input current to zero

    def set_input_current(self, current):
        self.neuron.I = current

    def stimulate(self, duration=100*ms):
        run(duration)
        spikes = self.spike_monitor.count[0]
        return spikes

# Database Manager Class
class DatabaseManager:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.db_name = f"session_{session_id}.db"
        try:
            self.conn = sqlite3.connect(self.db_name)
            self.create_tables()
        except sqlite3.Error as e:
            app_logger.error(f"Database connection failed: {e}")
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
            app_logger.error(f"Error saving agent to database: {e}")

    def save_conversation(self, agent_id: Optional[str], role: str, content: str) -> None:
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO conversation_history (session_id, agent_id, role, content)
                VALUES (?, ?, ?, ?)
            ''', (self.session_id, agent_id, role, content))
            self.conn.commit()
        except sqlite3.Error as e:
            app_logger.error(f"Error saving conversation: {e}")

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
            app_logger.error(f"Error loading conversation: {e}")
            return []

    def close(self) -> None:
        self.conn.close()

# Memory Module with STM and LTM
class MemoryModule:
    def __init__(self):
        self.short_term_memory = []
        self.long_term_memory = []
        self.replay_buffer = []
        self.quantum_memory = QuantumMemory(num_qubits=2)
        self.spiking_neuron = SpikingNeuron()

    def add_to_stm(self, data):
        self.short_term_memory.append(data)
        if len(self.short_term_memory) > 50:
            self.short_term_memory.pop(0)

    def consolidate_memory(self):
        if self.short_term_memory:
            # Create a superposition of short-term memories
            num_memories = len(self.short_term_memory)
            coefficients = np.array([1/np.sqrt(num_memories)] * num_memories, dtype=complex)
            memory_states = np.array(self.short_term_memory, dtype=complex)
            # Initialize quantum state as a superposition
            state_vector = np.sum(coefficients[:, np.newaxis] * memory_states, axis=0)
            self.quantum_memory.initialize_state(state_vector)
            # Apply an operator (e.g., identity)
            self.quantum_memory.apply_operator(lambda rho: rho)
            # Measure the quantum state
            collapsed_state = self.quantum_memory.measure()
            # Retrieve the consolidated memory
            consolidated_memory = self.short_term_memory[collapsed_state]
            self.long_term_memory.append(consolidated_memory)
            self.short_term_memory.clear()

    def recall_memory(self):
        if self.long_term_memory:
            self.spiking_neuron.set_input_current(1.5)  # Adjust the current as needed
            spikes = self.spiking_neuron.stimulate(duration=100*ms)
            if spikes > 0:
                # Recall the most recent long-term memory
                return self.long_term_memory[-1]
            else:
                return None
        else:
            return None

    def replay_experiences(self):
        for experience in self.replay_buffer:
            self.add_to_stm(experience)
        self.replay_buffer.clear()

# Base Agent Class with Collaborative Reasoning
class Agent:
    def __init__(
        self,
        name: str,
        role: str,
        backstory: str,
        style: str,
        instructions: str,
        model: str = "llama3.1:8b",
        db_manager: DatabaseManager = None,
        memory_module: MemoryModule = None
    ):
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
        self.memory_module = memory_module if memory_module else MemoryModule()

        if self.db_manager:
            self.db_manager.save_agent(self)

    def __repr__(self) -> str:
        return f"Agent(name={self.name}, role={self.role})"

    def set_agent_list(self, agent_list: List['Agent']) -> None:
        self.agent_list = agent_list
        self.prompt = self.create_prompt()

    def create_prompt(self) -> str:
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
            "Stay focused on the original how-to question and avoid going off-topic.\n"
        )
        return persona

    def respond(
        self,
        conversation_history: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        # Quantum-inspired decision-making
        qm = QuantumMemory(num_qubits=1)
        qm.initialize_state(np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex))
        # Apply an operator, e.g., Pauli-X gate
        operator = qml.matrix(qml.PauliX(wires=0))
        qm.apply_operator(lambda rho: operator @ rho @ operator.conj().T)
        decision = qm.measure()
        if decision == 0:
            # Collaborate with another agent
            if self.agent_list:
                partner_agent = random.choice(
                    [agent for agent in self.agent_list if agent.name != self.name]
                )
                shared_context = self.memory_module.recall_memory()
                partner_response = partner_agent.generate_response(shared_context)
                combined_response = self.generate_response(partner_response)
                response_content = combined_response
            else:
                response_content = self.generate_response()
        else:
            response_content = self.generate_response()

        # Save to conversation history
        if self.db_manager:
            self.db_manager.save_conversation(self.agent_id, self.name, response_content)
        return response_content

    def format_conversation_history(
        self,
        conversation_history: List[Dict[str, str]],
        limit: Optional[int] = None
    ) -> str:
        formatted_history = ""
        for entry in conversation_history[-limit:] if limit else conversation_history:
            role = entry['role']
            content = entry['content']
            formatted_history += f"{role}: {content}\n"
        return formatted_history

    def generate_response(self, input_text: Optional[str] = None):
        if input_text is None:
            input_text = self.prompt
        else:
            input_text = self.prompt + "\n" + input_text

        messages = [{'role': 'user', 'content': input_text}]
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={'temperature': 0.7, 'top_p': 0.9}
            )
            response_content = response['message']['content'].strip()
            self.memory_module.add_to_stm(response_content)
            self.memory_module.replay_buffer.append(response_content)
            return response_content
        except Exception as e:
            app_logger.error(f"Error generating response: {e}")
            return ""

# Guiding Agent Class to Keep Conversations On Topic
class GuidingAgent(Agent):
    def __init__(
        self,
        name: str,
        role: str,
        backstory: str,
        style: str,
        instructions: str,
        model: str = "llama3.1:8b",
        db_manager: DatabaseManager = None,
        memory_module: MemoryModule = None
    ):
        super().__init__(
            name, role, backstory, style, instructions, model, db_manager, memory_module
        )

    def respond(
        self,
        conversation_history: List[Dict[str, str]],
        original_question: str,
        temperature: float = 0.5,
        top_p: float = 0.7
    ) -> str:
        formatted_history = self.format_conversation_history(conversation_history)
        prompt = (
            f"You are {self.name}, {self.role}.\n"
            f"Backstory: {self.backstory}\n"
            f"Communication Style: {self.style}\n"
            f"{self.instructions}\n"
            "Your task is to ensure that all participants stay on topic related to the "
            "original how-to question.\n"
            "If you detect that the conversation is veering off-topic, politely remind "
            "the participants to focus on the main question.\n"
            f"Original Question: {original_question}\n\n"
            f"Conversation History:\n{formatted_history}\n\n"
            "Provide a gentle reminder to keep the discussion on track if necessary."
        )
        response_content = self.generate_response(prompt)
        if self.db_manager:
            self.db_manager.save_conversation(self.agent_id, self.name, response_content)
        return response_content

# State Evaluator Agent
class StateEvaluator:
    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model

    def evaluate(self, reasoning_chain: str) -> float:
        prompt = (
            "Evaluate the following reasoning for logic, coherence, and relevance to the "
            "problem.\n"
            f"Reasoning:\n{reasoning_chain}\n"
            "Score the reasoning on a scale from 0 to 1, where 1 is the highest score.\n"
            "Please provide the score first, followed by any additional comments.\n"
            "Score:"
        )
        messages = [{'role': 'user', 'content': prompt}]
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={'temperature': 0.0, 'top_p': 0.0}
            )
            response_content = response['message']['content'].strip()
            match = re.search(r"([0-1](?:\.\d+)?)", response_content)
            if match:
                score_str = match.group(1)
                score = float(score_str)
                return score
            else:
                return 0.0
        except Exception as e:
            app_logger.error(f"Error evaluating reasoning chain: {e}")
            return 0.0

# Thought Validator Agent
class ThoughtValidator:
    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model

    def validate(self, reasoning_chain: str) -> bool:
        prompt = (
            "As a Thought Validator, assess the following reasoning chain for logical "
            "consistency, factual accuracy, and completeness. Respond with 'Validated' if "
            "it passes all checks or 'Invalidated' if it fails any.\n"
            f"Reasoning Chain:\n{reasoning_chain}\n"
            "Validation Result:"
        )
        messages = [{'role': 'user', 'content': prompt}]
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={'temperature': 0.0, 'top_p': 0.0}
            )
            result = response['message']['content'].strip()
            return 'Validated' in result
        except Exception as e:
            app_logger.error(f"Error validating reasoning chain: {e}")
            return False

# Refinement Agent for Advanced Reasoning Phase
class RefinementAgent(Agent):
    def __init__(
        self,
        name: str,
        role: str,
        backstory: str,
        style: str,
        instructions: str,
        model: str = "llama3.1:8b",
        db_manager: DatabaseManager = None,
        memory_module: MemoryModule = None
    ):
        super().__init__(
            name, role, backstory, style, instructions, model, db_manager, memory_module
        )

    def respond(
        self,
        conversation_history: List[Dict[str, str]],
        original_question: str,
        temperature: float = 0.6,
        top_p: float = 0.8
    ) -> str:
        formatted_history = self.format_conversation_history(conversation_history)
        prompt = (
            f"You are {self.name}, {self.role}.\n"
            f"Backstory: {self.backstory}\n"
            f"Communication Style: {self.style}\n"
            f"{self.instructions}\n"
            f"Participants in the conversation: {', '.join([agent.name for agent in self.agent_list if agent.name != self.name])}.\n"
            "Your task is to refine the existing solutions by providing feedback, identifying "
            "potential improvements, and suggesting actionable enhancements.\n"
            "Respond in first person singular.\n"
            "Do not mention that you are an AI language model.\n"
            "Stay focused on refining the solutions related to the original how-to question.\n\n"
            f"Original Question: {original_question}\n\n"
            f"Conversation History:\n{formatted_history}\n\n"
            f"{self.name}:"
        )
        response_content = self.generate_response(prompt)
        if self.db_manager:
            self.db_manager.save_conversation(self.agent_id, self.name, response_content)
        return response_content

# Evaluation Agent for Advanced Reasoning Phase
class EvaluationAgent(Agent):
    def __init__(
        self,
        name: str,
        role: str,
        backstory: str,
        style: str,
        instructions: str,
        model: str = "llama3.1:8b",
        db_manager: DatabaseManager = None,
        memory_module: MemoryModule = None
    ):
        super().__init__(
            name, role, backstory, style, instructions, model, db_manager, memory_module
        )

    def respond(
        self,
        conversation_history: List[Dict[str, str]],
        original_question: str,
        temperature: float = 0.6,
        top_p: float = 0.8
    ) -> str:
        formatted_history = self.format_conversation_history(conversation_history)
        prompt = (
            f"You are {self.name}, {self.role}.\n"
            f"Backstory: {self.backstory}\n"
            f"Communication Style: {self.style}\n"
            f"{self.instructions}\n"
            f"Participants in the conversation: {', '.join([agent.name for agent in self.agent_list if agent.name != self.name])}.\n"
            "Your task is to evaluate the refined solutions for their feasibility, effectiveness, "
            "and practicality.\n"
            "Provide actionable feedback and suggest any necessary modifications to ensure "
            "successful implementation.\n"
            "Respond in first person singular.\n"
            "Do not mention that you are an AI language model.\n"
            "Stay focused on evaluating solutions related to the original how-to question.\n\n"
            f"Original Question: {original_question}\n\n"
            f"Conversation History:\n{formatted_history}\n\n"
            f"{self.name}:"
        )
        response_content = self.generate_response(prompt)
        if self.db_manager:
            self.db_manager.save_conversation(self.agent_id, self.name, response_content)
        return response_content

# Final Agent to Compile the How-To Manual
class FinalAgent:
    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model

    def compile_manual(self, topic: str, contributions: List[str], additional_text: str) -> str:
        prompt = (
            f"Using the following contributions from experts and the additional text provided, "
            f"write a comprehensive how-to manual on the topic.\n"
            f"Topic: {topic}\n"
            "Contributions:\n"
            + "\n\n".join(contributions) +
            "\n\nAdditional Text:\n" + additional_text +
            "\n\n"
            "Write the how-to manual in a clear and concise style with proper structure, including "
            "Introduction, Materials Needed, Step-by-Step Instructions, Tips and Tricks, and "
            "Conclusion.\n"
            "Ensure logical coherence and incorporate the key points from the contributions and the "
            "additional text.\n"
            "Do not mention the agents or the conversation.\n"
            "Focus on practical applications and actionable steps.\n"
            "Ensure each section is at least 1000 words long, with comprehensive details.\n"
            "Begin your how-to manual now."
        )
        messages = [{'role': 'user', 'content': prompt}]
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={'temperature': 0.7, 'top_p': 0.9, 'max_tokens': 2000}
            )
            manual = response['message']['content'].strip()
            return manual
        except Exception as e:
            app_logger.error(f"Error compiling how-to manual: {e}")
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
            "Analyze the following user input and suggest as many experts (up to 10, real "
            "non-fictional people) that could help solve the problem. Provide their names, roles, "
            "backstories, communication styles, and specific instructions for collaboration in the "
            "following JSON format:\n\n"
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
            "**Please output only the JSON data and nothing else. Ensure that all JSON syntax is "
            "correct, including commas between fields and objects.**"
        )
        messages = [{'role': 'user', 'content': prompt}]
        try:
            response = ollama.chat(
                model="llama3.1:8b",
                messages=messages,
                options={'temperature': 0.7, 'top_p': 0.9, 'max_tokens': 1000}
            )
            suggested_agents = response['message']['content'].strip()
            self.parse_and_create_agents(suggested_agents)
        except Exception as e:
            app_logger.error(f"Error analyzing input: {e}")

    def parse_and_create_agents(self, suggested_agents_text: str) -> None:
        agents = []
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                agent_list = json.loads(suggested_agents_text)
                for agent_info in agent_list:
                    agent = Agent(
                        name=agent_info.get('Name', 'Unknown'),
                        role=agent_info.get('Role', ''),
                        backstory=agent_info.get('Backstory', ''),
                        style=agent_info.get('Style', ''),
                        instructions=agent_info.get('Instructions', 'Collaborate effectively.'),
                        model="llama3.1:8b",
                        db_manager=self.db_manager,
                        memory_module=MemoryModule()
                    )
                    agents.append(agent)
                if not agents:
                    app_logger.error("No agents were parsed from the LLM's response.")
                else:
                    for agent in agents:
                        agent.set_agent_list(agents)
                        app_logger.info(f"Agent '{agent.name}' has been created.")
                    self.agents = agents

                    guiding_agent_info = {
                        "Name": "GuidingAgent",
                        "Role": "Conversation Moderator",
                        "Backstory": "You are a seasoned moderator trained to keep discussions "
                                     "focused and productive.",
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
                        db_manager=self.db_manager,
                        memory_module=MemoryModule()
                    )
                    self.guiding_agent = guiding_agent
                    self.agents.append(guiding_agent)
                    app_logger.info(f"GuidingAgent '{guiding_agent.name}' has been created.")
                    break
            except json.JSONDecodeError as e:
                app_logger.error(f"JSON decoding error: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    suggested_agents_text = self.correct_json(suggested_agents_text)
                else:
                    app_logger.error("Maximum retries reached. Could not parse agents.")
                    print("Error: Unable to parse agent information. Please try again later.")
                    return
            except Exception as e:
                app_logger.error(f"Error parsing agents: {e}")
                print("Error: Unable to parse agent information. Please try again later.")
                return

    def correct_json(self, json_text: str) -> str:
        json_text = re.sub(r'}\s*{', r'}, {', json_text)
        json_text = re.sub(r'"\s*"Style', r'", "Style', json_text)
        json_text = re.sub(r'("Instructions": ".*?")\s*}', r'\1}', json_text)
        return json_text

    def run_collaborative_reasoning(self, message: str) -> Optional[str]:
        if self.phase == "PAST":
            self.iteration_count += 1
            app_logger.info(f"--- Phase 1: Initial Spawning (PAST) ---\n")
            combined_conversation = self.conversation_history.copy()
            contributions = []
            reasoning_branches: Dict[str, str] = {}
            validated_branches: Dict[str, str] = {}

            # Each agent contributes based on their defined role.
            for agent in self.agents:
                if isinstance(agent, GuidingAgent):
                    reminder = agent.respond(combined_conversation, self.original_question)
                    combined_conversation.append({'role': agent.name, 'content': reminder})
                    self.conversation_history.append({'role': agent.name, 'content': reminder})
                    app_logger.info(f"{agent.name} provided a reminder to stay on topic.\n")
                    continue

                app_logger.info(f"{agent.name} is generating response...\n")
                agent_response = agent.respond(combined_conversation)
                combined_conversation.append({'role': agent.name, 'content': agent_response})
                self.conversation_history.append({'role': agent.name, 'content': agent_response})
                contributions.append(agent_response)
                reasoning_branches[agent.name] = agent_response

                if self.guiding_agent:
                    reminder = self.guiding_agent.respond(combined_conversation, self.original_question)
                    if reminder:
                        combined_conversation.append({'role': self.guiding_agent.name, 'content': reminder})
                        self.conversation_history.append({'role': self.guiding_agent.name, 'content': reminder})
                        app_logger.info(f"{self.guiding_agent.name} provided a reminder to stay on topic.\n")

            # Validation and Thought Evaluation (RAFT Method)
            for agent_name, reasoning_chain in reasoning_branches.items():
                is_valid = self.thought_validator.validate(reasoning_chain)
                if is_valid:
                    validated_branches[agent_name] = reasoning_chain
                    app_logger.info(f"{agent_name}'s reasoning is Validated.\n")
                else:
                    app_logger.info(f"{agent_name}'s reasoning is Invalidated.\n")

            # Proceed to compile how-to manual if any branches are validated
            if validated_branches:
                valid_contributions = [reasoning_chain for reasoning_chain in validated_branches.values()]
                final_manual = self.final_agent.compile_manual(
                    self.original_question, valid_contributions, self.additional_text
                )
                if final_manual:
                    self.save_final_solution(final_manual)
                    self.phase = "ADVANCED_REASONING"
                    return final_manual
                else:
                    app_logger.info("Unable to compile final how-to manual.\n")
                    return None
            else:
                app_logger.info("No validated reasoning branches. Cannot produce a final how-to manual.\n")
                return None

        elif self.phase == "ADVANCED_REASONING":
            self.iteration_count += 1
            app_logger.info(f"--- Phase 2: Advanced Reasoning (RAFT & EAT) ---\n")
            combined_conversation = self.conversation_history.copy()
            contributions = []
            reasoning_branches: Dict[str, str] = {}
            validated_branches: Dict[str, str] = {}

            # Refinement and Evaluation Agents if not already created
            if not self.refinement_agent:
                refinement_agent_info = {
                    "Name": "RefinementAgent",
                    "Role": "Refinement Specialist",
                    "Backstory": "You are an expert in refining and improving solutions through "
                                 "structured feedback.",
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
                    db_manager=self.db_manager,
                    memory_module=MemoryModule()
                )
                self.agents.append(self.refinement_agent)
                app_logger.info(f"RefinementAgent '{self.refinement_agent.name}' has been created.")

            if not self.evaluation_agent:
                evaluation_agent_info = {
                    "Name": "EvaluationAgent",
                    "Role": "Solution Evaluator",
                    "Backstory": "You are an expert in evaluating the feasibility and effectiveness "
                                 "of solutions.",
                    "Style": "Critical and thorough.",
                    "Instructions": "Assess the solutions for their practicality and suggest necessary "
                                    "modifications."
                }
                self.evaluation_agent = EvaluationAgent(
                    name=evaluation_agent_info["Name"],
                    role=evaluation_agent_info["Role"],
                    backstory=evaluation_agent_info["Backstory"],
                    style=evaluation_agent_info["Style"],
                    instructions=evaluation_agent_info["Instructions"],
                    model="llama3.1:8b",
                    db_manager=self.db_manager,
                    memory_module=MemoryModule()
                )
                self.agents.append(self.evaluation_agent)
                app_logger.info(f"EvaluationAgent '{self.evaluation_agent.name}' has been created.")

            # Each agent contributes during the advanced reasoning phase.
            for agent in self.agents:
                if isinstance(agent, (GuidingAgent, RefinementAgent, EvaluationAgent)):
                    if isinstance(agent, RefinementAgent):
                        refinement_response = agent.respond(combined_conversation, self.original_question)
                        combined_conversation.append({'role': agent.name, 'content': refinement_response})
                        self.conversation_history.append({'role': agent.name, 'content': refinement_response})
                        contributions.append(refinement_response)
                        reasoning_branches[agent.name] = refinement_response
                        app_logger.info(f"{agent.name} provided refinement suggestions.\n")
                    elif isinstance(agent, EvaluationAgent):
                        evaluation_response = agent.respond(combined_conversation, self.original_question)
                        combined_conversation.append({'role': agent.name, 'content': evaluation_response})
                        self.conversation_history.append({'role': agent.name, 'content': evaluation_response})
                        contributions.append(evaluation_response)
                        reasoning_branches[agent.name] = evaluation_response
                        app_logger.info(f"{agent.name} provided evaluation feedback.\n")
                    else:
                        reminder = agent.respond(combined_conversation, self.original_question)
                        combined_conversation.append({'role': agent.name, 'content': reminder})
                        self.conversation_history.append({'role': agent.name, 'content': reminder})
                        app_logger.info(f"{agent.name} provided a reminder to stay on topic.\n")
                    continue

                # For other agents (not GuidingAgent, RefinementAgent, EvaluationAgent)
                app_logger.info(f"{agent.name} is refining their response...\n")
                agent_response = agent.respond(combined_conversation)
                combined_conversation.append({'role': agent.name, 'content': agent_response})
                self.conversation_history.append({'role': agent.name, 'content': agent_response})
                contributions.append(agent_response)
                reasoning_branches[agent.name] = agent_response

                if self.guiding_agent:
                    reminder = self.guiding_agent.respond(combined_conversation, self.original_question)
                    if reminder:
                        combined_conversation.append({'role': self.guiding_agent.name, 'content': reminder})
                        self.conversation_history.append({'role': self.guiding_agent.name, 'content': reminder})
                        app_logger.info(f"{self.guiding_agent.name} provided a reminder to stay on topic.\n")

            # RAFT Process for Validation and Feedback
            for agent_name, reasoning_chain in reasoning_branches.items():
                is_valid = self.thought_validator.validate(reasoning_chain)
                if is_valid:
                    validated_branches[agent_name] = reasoning_chain
                    app_logger.info(f"{agent_name}'s reasoning is Validated.\n")
                else:
                    app_logger.info(f"{agent_name}'s reasoning is Invalidated.\n")

            # Compile final how-to manual if there are validated branches
            if validated_branches:
                valid_contributions = [reasoning_chain for reasoning_chain in validated_branches.values()]
                final_manual = self.final_agent.compile_manual(
                    self.original_question, valid_contributions, self.additional_text
                )
                if final_manual:
                    self.save_final_solution(final_manual)
                    return final_manual
                else:
                    app_logger.info("Unable to compile final how-to manual.\n")
                    return None
            else:
                app_logger.info("No validated reasoning branches. Cannot produce a final how-to manual.\n")
                return None

    def save_final_solution(self, solution: str) -> None:
        if self.original_question:
            filename = f"HOWTO.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(solution)
            app_logger.info(f"Final how-to manual saved to {filename}")
        else:
            app_logger.error("Original question not found. Cannot save final solution.")

    def close_session(self) -> None:
        self.db_manager.close()

    def visualize_thought_process(self, step: str) -> None:
        print(f"Visualization: {step}")
        app_logger.info(f"Visualization: {step}")

    def start_chat(self) -> None:
        print("Welcome to the Legion AI multi-agent spawning and reasoning system.")
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
                    self.additional_text += f"\n{user_input}"
                    self.visualize_thought_process("Step 3: Agents are brainstorming solutions...")

                print("Please provide any additional text or references you would like to include (or press Enter to skip):")
                additional_input = input().strip()
                if additional_input:
                    self.additional_text += f"\n{additional_input}"

                self.conversation_history.append({'role': 'User', 'content': user_input})
                self.db_manager.save_conversation(None, 'User', user_input)

                self.visualize_thought_process("Step 4: Agents are refining their thoughts...")

                final_manual = self.run_collaborative_reasoning(user_input)

                if final_manual:
                    print(f"Final How-To Manual:\n{final_manual}\n")
                else:
                    print("Unable to produce a final how-to manual.\n")

# Main Function
def main():
    chat_manager = ChatManager()
    chat_manager.start_chat()

if __name__ == "__main__":
    main()

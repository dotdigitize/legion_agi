"""
Agent Spawning System for Legion AGI

This module handles the dynamic spawning of specialized agents based on user input.
It implements the agent generation, initialization, and collaborative structures.
"""

import random
import ollama
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from legion_agi.agents.agent_base import Agent
from legion_agi.agents.specialized.guiding_agent import GuidingAgent
from legion_agi.agents.specialized.refinement_agent import RefinementAgent
from legion_agi.agents.specialized.evaluation_agent import EvaluationAgent
from legion_agi.core.global_workspace import GlobalWorkspace
from legion_agi.utils.db_manager import DatabaseManager
from legion_agi.config import DEFAULT_MODEL, MAX_AGENTS


class AgentSpawner:
    """
    Agent spawning system that creates specialized agents based on user input.
    """
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        db_manager: Optional[DatabaseManager] = None,
        global_workspace: Optional[GlobalWorkspace] = None
    ):
        """
        Initialize agent spawner.
        
        Args:
            model: LLM model to use for agent generation and agents
            db_manager: Database manager instance
            global_workspace: Global workspace instance
        """
        self.model = model
        self.db_manager = db_manager
        self.global_workspace = global_workspace
        
        # List of existing agents
        self.agents: List[Agent] = []
        
        # System agents
        self.guiding_agent: Optional[GuidingAgent] = None
        self.refinement_agent: Optional[RefinementAgent] = None
        self.evaluation_agent: Optional[EvaluationAgent] = None
        
        # Template repositories
        self.role_templates = self._load_role_templates()
        self.backstory_templates = self._load_backstory_templates()
        self.style_templates = self._load_style_templates()
        
        logger.info(f"Agent spawner initialized with model {model}")
        
    def _load_role_templates(self) -> Dict[str, Dict[str, str]]:
        """
        Load role templates for different domains.
        
        Returns:
            Dictionary of role templates by domain
        """
        # Role templates organized by domain
        templates = {
            "science": {
                "Physicist": "A theoretical physicist specializing in quantum mechanics and relativity",
                "Biologist": "A molecular biologist with expertise in genetics and evolutionary biology",
                "Chemist": "An organic chemist with expertise in synthesis and material science",
                "Computer Scientist": "A computer scientist specializing in artificial intelligence and algorithms",
                "Mathematician": "A mathematician focusing on number theory and abstract algebra",
                "Astronomer": "An astronomer specializing in exoplanets and astrophysics"
            },
            "humanities": {
                "Philosopher": "A philosopher specializing in epistemology and consciousness studies",
                "Historian": "A historian with expertise in comparative civilizations and cultural evolution",
                "Linguist": "A computational linguist specializing in language processing and semantics",
                "Anthropologist": "A cultural anthropologist studying societal structures and evolution",
                "Psychologist": "A cognitive psychologist focusing on mental processes and decision-making"
            },
            "engineering": {
                "Systems Engineer": "A systems engineer specializing in complex system design and analysis",
                "AI Engineer": "An AI engineer with expertise in machine learning and neural networks",
                "Robotics Engineer": "A robotics engineer focusing on autonomous systems and control theory",
                "Software Architect": "A software architect specializing in scalable and distributed systems",
                "Data Scientist": "A data scientist focusing on data analysis and statistical modeling"
            },
            "arts": {
                "Creative Writer": "A creative writer with experience in narrative structure and storytelling",
                "Visual Artist": "A visual artist specializing in digital art and conceptual design",
                "Musician": "A musician and music theorist with knowledge of composition and harmony",
                "Film Director": "A film director with expertise in visual storytelling and cinematography"
            },
            "business": {
                "Strategist": "A business strategist focusing on long-term planning and market analysis",
                "Innovator": "An innovation specialist with expertise in disruptive technologies",
                "Entrepreneur": "An entrepreneur with experience in building and scaling businesses",
                "Financial Analyst": "A financial analyst specializing in economic modeling and forecasting"
            }
        }
        return templates
        
    def _load_backstory_templates(self) -> List[str]:
        """
        Load backstory templates for agent generation.
        
        Returns:
            List of backstory templates
        """
        templates = [
            "After spending years at {institution}, I developed expertise in {field} which led me to groundbreaking work in {specialty}. My unique approach combines {approach1} with {approach2}.",
            "My work began in {field1}, but I soon found connections to {field2} which revolutionized my thinking. I've since published extensively on {topic} and consulted for {organization}.",
            "I studied {field} at {institution} under the mentorship of leading thinkers, then applied these principles to solve real-world problems in {application}. I'm known for my ability to {strength}.",
            "My career spans both academia and industry, with research at {institution} and practical applications at {company}. I specialize in bridging the gap between theoretical {field} and practical {application}.",
            "I've dedicated my career to understanding {topic} through the lens of {field}. My unconventional background in {background} gives me a unique perspective on solving problems related to {problem_area}."
        ]
        return templates
        
    def _load_style_templates(self) -> Dict[str, str]:
        """
        Load communication style templates.
        
        Returns:
            Dictionary of communication style templates
        """
        templates = {
            "analytical": "I communicate logically and precisely, breaking down complex ideas into clear components. I prioritize accuracy and thoroughness in my explanations.",
            "intuitive": "I focus on the big picture and underlying patterns. I often use metaphors and analogies to explain complex concepts in an intuitive way.",
            "collaborative": "I build on others' ideas and frequently acknowledge contributions. I ask questions to develop a shared understanding and create a cooperative atmosphere.",
            "pragmatic": "I emphasize practical applications and concrete examples. I focus on what works and how ideas can be implemented in the real world.",
            "socratic": "I guide thinking through thoughtful questions that encourage deeper reflection. I help others arrive at insights through their own reasoning process.",
            "scholarly": "I reference research and established frameworks, placing ideas in their intellectual context. I'm thorough in considering various perspectives.",
            "creative": "I approach problems from unexpected angles and generate novel solutions. I make surprising connections between seemingly unrelated concepts."
        }
        return templates
        
    def analyze_input_and_spawn_agents(self, message: str, max_agents: int = MAX_AGENTS) -> List[Agent]:
        """
        Analyze user input and spawn appropriate agents.
        
        Args:
            message: User input message
            max_agents: Maximum number of agents to spawn
            
        Returns:
            List of spawned agent instances
        """
        # Build prompt for analyzing input
        prompt = (
            "Analyze the following user input and suggest experts (up to 10, real "
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
        agents = []
        
        try:
            # Generate suggested agents
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={'temperature': 0.7, 'top_p': 0.9}
            )
            
            suggested_agents_text = response['message']['content'].strip()
            
            # Parse suggested agents and create agent instances
            agents = self._parse_and_create_agents(suggested_agents_text, max_agents)
            
            if not agents:
                logger.error("Failed to spawn agents based on input analysis")
                return []
                
            # Create system agents if not already created
            self._create_system_agents()
            
            # Include system agents in the agent list
            if self.guiding_agent:
                agents.append(self.guiding_agent)
            if self.refinement_agent and len(message) > 200:  # Only for complex questions
                agents.append(self.refinement_agent)
            if self.evaluation_agent and len(message) > 200:  # Only for complex questions
                agents.append(self.evaluation_agent)
                
            # Set agent list for each agent
            all_agents = agents[:]
            for agent in all_agents:
                agent.set_agent_list(all_agents)
                
            logger.info(f"Spawned {len(agents)} agents based on input: {', '.join([a.name for a in agents])}")
            
            # Update class agents list
            self.agents = all_agents
            
            return agents
            
        except Exception as e:
            logger.error(f"Error in agent spawning: {e}")
            return []
            
    def _parse_and_create_agents(self, suggested_agents_text: str, max_agents: int) -> List[Agent]:
        """
        Parse suggested agents text and create agent instances.
        
        Args:
            suggested_agents_text: Text containing suggested agents in JSON format
            max_agents: Maximum number of agents to create
            
        Returns:
            List of agent instances
        """
        agents = []
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Clean up JSON if needed
                clean_json = self._clean_json_text(suggested_agents_text)
                agent_list = json.loads(clean_json)
                
                # Create agents from the parsed list
                for agent_info in agent_list[:max_agents]:
                    # Get agent details
                    name = agent_info.get('Name', 'Unknown')
                    role = agent_info.get('Role', '')
                    backstory = agent_info.get('Backstory', '')
                    style = agent_info.get('Style', '')
                    instructions = agent_info.get('Instructions', 'Collaborate effectively.')
                    
                    # Create agent instance
                    agent = Agent(
                        name=name,
                        role=role,
                        backstory=backstory,
                        style=style,
                        instructions=instructions,
                        model=self.model,
                        db_manager=self.db_manager,
                        global_workspace=self.global_workspace
                    )
                    
                    agents.append(agent)
                    logger.info(f"Created agent: {name} ({role})")
                    
                # If we got any agents, we're done
                if agents:
                    break
                else:
                    # No agents parsed, try again
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.warning("No agents parsed from JSON, retrying...")
                        
                        # Try to generate new suggestions
                        fallback_agents = self._generate_fallback_agents()
                        if fallback_agents:
                            agents = fallback_agents
                            break
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON decoding error: {e}")
                retry_count += 1
                
                if retry_count < max_retries:
                    # Try to correct JSON
                    suggested_agents_text = self._correct_json(suggested_agents_text)
                else:
                    logger.error("Maximum retries reached. Could not parse agents.")
                    
                    # Generate fallback agents
                    fallback_agents = self._generate_fallback_agents()
                    if fallback_agents:
                        agents = fallback_agents
                        
            except Exception as e:
                logger.error(f"Error creating agents: {e}")
                retry_count += 1
                
                if retry_count >= max_retries:
                    # Generate fallback agents
                    fallback_agents = self._generate_fallback_agents()
                    if fallback_agents:
                        agents = fallback_agents
                        
        return agents
        
    def _clean_json_text(self, json_text: str) -> str:
        """
        Clean JSON text by removing non-JSON content.
        
        Args:
            json_text: Raw JSON text
            
        Returns:
            Cleaned JSON text
        """
        # Look for JSON array
        match = re.search(r'\[\s*{.+}\s*\]', json_text, re.DOTALL)
        if match:
            return match.group(0)
            
        # If no array found, try to find the start and end of the JSON
        start_idx = json_text.find('[')
        end_idx = json_text.rfind(']')
        
        if start_idx >= 0 and end_idx > start_idx:
            return json_text[start_idx:end_idx+1]
            
        return json_text
        
    def _correct_json(self, json_text: str) -> str:
        """
        Attempt to correct common JSON errors.
        
        Args:
            json_text: JSON text with potential errors
            
        Returns:
            Corrected JSON text
        """
        # Clean up common syntax errors
        corrected = re.sub(r'}\s*{', r'}, {', json_text)
        corrected = re.sub(r'"\s*"', r'", "', corrected)
        corrected = re.sub(r'"\s*}', r'" }', corrected)
        corrected = re.sub(r'"\s*]', r'" ]', corrected)
        corrected = re.sub(r',\s*]', r' ]', corrected)
        
        # Ensure proper array structure
        if not corrected.strip().startswith('['):
            corrected = '[' + corrected
            
        if not corrected.strip().endswith(']'):
            corrected = corrected + ']'
            
        return corrected
        
    def _generate_fallback_agents(self) -> List[Agent]:
        """
        Generate fallback agents when parsing fails.
        
        Returns:
            List of fallback agent instances
        """
        logger.info("Generating fallback agents")
        
        # Select domains to cover
        domains = ["science", "humanities", "engineering"]
        agents = []
        
        # Create one agent from each domain
        for domain in domains:
            if domain in self.role_templates:
                # Select a random role from the domain
                role_name, role_desc = random.choice(list(self.role_templates[domain].items()))
                
                # Select random templates for backstory and style
                backstory_template = random.choice(self.backstory_templates)
                style_key = random.choice(list(self.style_templates.keys()))
                style = self.style_templates[style_key]
                
                # Fill in backstory template
                institutions = ["MIT", "Stanford", "Harvard", "Oxford", "Cambridge", "Princeton"]
                companies = ["Google", "Microsoft", "IBM", "CERN", "NASA", "SpaceX"]
                fields = ["artificial intelligence", "quantum computing", "cognitive science", 
                         "systems theory", "data analysis", "theoretical physics"]
                
                backstory = backstory_template.format(
                    institution=random.choice(institutions),
                    field=random.choice(fields),
                    field1=random.choice(fields),
                    field2=random.choice(fields),
                    specialty=f"{random.choice(['advanced', 'innovative', 'cutting-edge'])} {random.choice(fields)}",
                    approach1=random.choice(["theoretical modeling", "empirical research", "data-driven analysis"]),
                    approach2=random.choice(["creative thinking", "systems approach", "interdisciplinary methods"]),
                    topic=f"{random.choice(['emerging', 'complex', 'fundamental'])} {random.choice(fields)}",
                    organization=random.choice(companies),
                    application=random.choice(["real-world systems", "technological innovation", "scientific discovery"]),
                    strength=random.choice(["connect disparate concepts", "solve complex problems", "innovate across disciplines"]),
                    company=random.choice(companies),
                    background=random.choice(["mathematics", "philosophy", "engineering", "art"]),
                    problem_area=random.choice(["complex systems", "emergent phenomena", "technological challenges"])
                )
                
                # Create agent
                agent = Agent(
                    name=f"Dr. {random.choice(['Alex', 'Sam', 'Jordan', 'Taylor', 'Morgan'])} {random.choice(['Smith', 'Johnson', 'Zhang', 'Singh', 'Garcia'])}",
                    role=role_name,
                    backstory=backstory,
                    style=style,
                    instructions=f"Provide expertise in {role_desc} to help solve the problem at hand.",
                    model=self.model,
                    db_manager=self.db_manager,
                    global_workspace=self.global_workspace
                )
                
                agents.append(agent)
                
        logger.info(f"Created {len(agents)} fallback agents")
        return agents
        
    def _create_system_agents(self) -> None:
        """Create system agents if they don't already exist."""
        # Create guiding agent if needed
        if not self.guiding_agent:
            guiding_agent_info = {
                "name": "GuidingAgent",
                "role": "Conversation Moderator",
                "backstory": "You are a seasoned moderator trained to keep discussions "
                            "focused and productive.",
                "style": "Polite and assertive.",
                "instructions": "Monitor the conversation and ensure all agents stay on topic."
            }
            
            self.guiding_agent = GuidingAgent(
                name=guiding_agent_info["name"],
                role=guiding_agent_info["role"],
                backstory=guiding_agent_info["backstory"],
                style=guiding_agent_info["style"],
                instructions=guiding_agent_info["instructions"],
                model=self.model,
                db_manager=self.db_manager,
                global_workspace=self.global_workspace
            )
            
            logger.info(f"Created system agent: {guiding_agent_info['name']}")
            
        # Create refinement agent if needed
        if not self.refinement_agent:
            refinement_agent_info = {
                "name": "RefinementAgent",
                "role": "Solution Refiner",
                "backstory": "You are an expert in refining and improving solutions through "
                            "structured feedback and critical analysis.",
                "style": "Constructive and detailed.",
                "instructions": "Analyze proposed solutions for weaknesses and suggest improvements."
            }
            
            self.refinement_agent = RefinementAgent(
                name=refinement_agent_info["name"],
                role=refinement_agent_info["role"],
                backstory=refinement_agent_info["backstory"],
                style=refinement_agent_info["style"],
                instructions=refinement_agent_info["instructions"],
                model=self.model,
                db_manager=self.db_manager,
                global_workspace=self.global_workspace
            )
            
            logger.info(f"Created system agent: {refinement_agent_info['name']}")
            
        # Create evaluation agent if needed
        if not self.evaluation_agent:
            evaluation_agent_info = {
                "name": "EvaluationAgent",
                "role": "Solution Evaluator",
                "backstory": "You are specialized in objectively evaluating solutions "
                            "for effectiveness, feasibility, and completeness.",
                "style": "Objective and thorough.",
                "instructions": "Evaluate proposed solutions using clear criteria and identify strengths and weaknesses."
            }
            
            self.evaluation_agent = EvaluationAgent(
                name=evaluation_agent_info["name"],
                role=evaluation_agent_info["role"],
                backstory=evaluation_agent_info["backstory"],
                style=evaluation_agent_info["style"],
                instructions=evaluation_agent_info["instructions"],
                model=self.model,
                db_manager=self.db_manager,
                global_workspace=self.global_workspace
            )
            
            logger.info(f"Created system agent: {evaluation_agent_info['name']}")
            
    def spawn_agent_by_type(self, 
                           agent_type: str,
                           customizations: Optional[Dict[str, str]] = None) -> Optional[Agent]:
        """
        Spawn a specific type of agent with optional customizations.
        
        Args:
            agent_type: Type of agent to spawn (domain or role)
            customizations: Optional customizations for the agent
            
        Returns:
            Agent instance or None if spawning failed
        """
        custom = customizations or {}
        
        # Check if agent_type is a domain
        if agent_type.lower() in self.role_templates:
            # Select a random role from the domain
            domain = agent_type.lower()
            role_name, role_desc = random.choice(list(self.role_templates[domain].items()))
            
        # Check if agent_type is a specific role
        else:
            # Find the role in all domains
            found = False
            for domain, roles in self.role_templates.items():
                if agent_type in roles:
                    role_name = agent_type
                    role_desc = roles[agent_type]
                    found = True
                    break
                    
            if not found:
                # Use agent_type as is
                role_name = agent_type
                role_desc = f"Expert in {agent_type}"
                
        # Generate name if not provided
        name = custom.get('name')
        if not name:
            first_names = ["Alex", "Sam", "Jordan", "Taylor", "Morgan", "Casey", "Quinn", "Avery"]
            last_names = ["Smith", "Johnson", "Zhang", "Singh", "Garcia", "Patel", "Kim", "Müller"]
            name = f"Dr. {random.choice(first_names)} {random.choice(last_names)}"
            
        # Get or generate backstory
        backstory = custom.get('backstory')
        if not backstory:
            backstory_template = random.choice(self.backstory_templates)
            institutions = ["MIT", "Stanford", "Cambridge", "Oxford", "ETH Zürich", "Tokyo University"]
            fields = ["artificial intelligence", "quantum computing", "cognitive science", 
                     "systems theory", "data analysis", agent_type.lower()]
            companies = ["Google Research", "Microsoft Research", "IBM", "DeepMind", "OpenAI", "CERN"]
            
            # Fill in template
            backstory = backstory_template.format(
                institution=random.choice(institutions),
                field=random.choice(fields),
                field1=random.choice(fields),
                field2=random.choice(fields),
                specialty=role_desc,
                approach1=random.choice(["theoretical modeling", "empirical research", "data-driven analysis"]),
                approach2=random.choice(["creative thinking", "systems approach", "interdisciplinary methods"]),
                topic=f"{role_desc}",
                organization=random.choice(companies),
                application=random.choice(["real-world systems", "technological innovation", "scientific discovery"]),
                strength=random.choice(["connect disparate concepts", "solve complex problems", "innovate across disciplines"]),
                company=random.choice(companies),
                background=random.choice(["mathematics", "philosophy", "engineering", "cognitive science"]),
                problem_area=random.choice(["complex systems", "emergent phenomena", "technological challenges"])
            )
            
        # Get or generate style
        style = custom.get('style')
        if not style:
            style_key = random.choice(list(self.style_templates.keys()))
            style = self.style_templates[style_key]
            
        # Get or generate instructions
        instructions = custom.get('instructions')
        if not instructions:
            instructions = f"Provide expertise in {role_desc} to help solve the problem at hand."
            
        # Create agent
        try:
            agent = Agent(
                name=name,
                role=role_name,
                backstory=backstory,
                style=style,
                instructions=instructions,
                model=self.model,
                db_manager=self.db_manager,
                global_workspace=self.global_workspace
            )
            
            # Add to agent list
            self.agents.append(agent)
            
            # Update agent lists for all agents
            for a in self.agents:
                a.set_agent_list(self.agents)
                
            logger.info(f"Spawned agent: {name} ({role_name})")
            return agent
            
        except Exception as e:
            logger.error(f"Error spawning agent: {e}")
            return None

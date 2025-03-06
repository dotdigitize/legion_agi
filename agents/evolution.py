"""
Agent Evolution System for Legion AGI

This module implements the agent evolution mechanism, allowing agents to evolve
over time through simulated evolutionary processes including selection,
crossover, and mutation of cognitive traits and knowledge.
"""

import numpy as np
import random
import uuid
import datetime
import json
import os
from typing import List, Dict, Tuple, Any, Optional, Union, Set
from loguru import logger

from legion_agi.agents.agent_base import Agent
from legion_agi.agents.spawning import AgentSpawner
from legion_agi.core.memory_module import MemoryModule
from legion_agi.utils.db_manager import DatabaseManager
from legion_agi.config import (
    MUTATION_RATE,
    CROSSOVER_RATE,
    SELECTION_PRESSURE,
    GENERATIONS_PER_CYCLE
)


class AgentEvolution:
    """
    Agent evolution system for Legion AGI.
    Implements agent population management and evolutionary processes.
    """
    
    def __init__(
        self,
        agent_spawner: AgentSpawner,
        db_manager: Optional[DatabaseManager] = None,
        population_size: int = 10,
        evolution_data_dir: str = "evolution_data"
    ):
        """
        Initialize agent evolution system.
        
        Args:
            agent_spawner: Agent spawner for creating new agents
            db_manager: Database manager for persistence
            population_size: Maximum population size
            evolution_data_dir: Directory for evolution data
        """
        self.agent_spawner = agent_spawner
        self.db_manager = db_manager
        self.population_size = population_size
        self.evolution_data_dir = evolution_data_dir
        
        # Ensure evolution data directory exists
        os.makedirs(evolution_data_dir, exist_ok=True)
        
        # Population management
        self.population: List[Agent] = []
        self.generation: int = 0
        self.fitness_scores: Dict[str, float] = {}
        self.evolution_history: List[Dict[str, Any]] = []
        
        # Fitness evaluation cache
        self.fitness_cache: Dict[str, Dict[str, float]] = {}
        
        # Evolution parameters
        self.mutation_rate = MUTATION_RATE
        self.crossover_rate = CROSSOVER_RATE
        self.selection_pressure = SELECTION_PRESSURE
        
        logger.info(f"Agent evolution system initialized with population size {population_size}")
        
    def initialize_population(self, initial_agents: Optional[List[Agent]] = None) -> None:
        """
        Initialize agent population.
        
        Args:
            initial_agents: Optional list of initial agents
        """
        # Start with provided agents if available
        if initial_agents:
            self.population = initial_agents[:]
            
        # Fill remaining population slots
        remaining = self.population_size - len(self.population)
        if remaining > 0:
            # Create agents for different domains to ensure diversity
            domains = ["science", "humanities", "engineering", "arts", "business"]
            
            for i in range(remaining):
                domain = domains[i % len(domains)]
                agent = self.agent_spawner.spawn_agent_by_type(domain)
                
                if agent:
                    self.population.append(agent)
                    
        # Initialize fitness scores
        for agent in self.population:
            self.fitness_scores[agent.agent_id] = 0.5  # Default neutral score
            
        # Record initial population
        self._record_generation()
        
        logger.info(f"Initialized population with {len(self.population)} agents")
        
    def evolve_population(self, num_generations: int = GENERATIONS_PER_CYCLE) -> List[Agent]:
        """
        Evolve the agent population through multiple generations.
        
        Args:
            num_generations: Number of generations to evolve
            
        Returns:
            List of evolved agents
        """
        for i in range(num_generations):
            logger.info(f"Starting evolution generation {self.generation + 1}")
            
            # Evaluate fitness if needed
            if not all(agent.agent_id in self.fitness_scores for agent in self.population):
                self.evaluate_population_fitness()
                
            # Select parents
            parents = self._select_parents()
            
            # Create offspring
            offspring = self._create_offspring(parents)
            
            # Apply mutations
            mutated_offspring = self._mutate_offspring(offspring)
            
            # Select survivors (elitism + offspring)
            self._select_survivors(mutated_offspring)
            
            # Update generation counter
            self.generation += 1
            
            # Record generation data
            self._record_generation()
            
            logger.info(f"Completed evolution generation {self.generation}")
            
        return self.population
        
    def evaluate_population_fitness(self, 
                                   problem_set: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate fitness of all agents in the population.
        
        Args:
            problem_set: Optional list of problems to evaluate agents on
            
        Returns:
            Dictionary mapping agent IDs to fitness scores
        """
        # Generate problem set if not provided
        if not problem_set:
            problem_set = self._generate_problem_set()
            
        logger.info(f"Evaluating population fitness on {len(problem_set)} problems")
        
        # For each agent, evaluate on problem set
        for agent in self.population:
            agent_id = agent.agent_id
            
            # Skip if already evaluated
            if agent_id in self.fitness_scores:
                continue
                
            # Calculate fitness score
            fitness = self._evaluate_agent_fitness(agent, problem_set)
            self.fitness_scores[agent_id] = fitness
            
            logger.debug(f"Agent {agent.name} fitness: {fitness}")
            
        return self.fitness_scores
        
    def _evaluate_agent_fitness(self, agent: Agent, problem_set: List[str]) -> float:
        """
        Evaluate fitness of a single agent on a problem set.
        
        Args:
            agent: Agent to evaluate
            problem_set: List of problems to evaluate on
            
        Returns:
            Fitness score (0.0 to 1.0)
        """
        # Check cache first
        if agent.agent_id in self.fitness_cache:
            # For each problem, check if it's in the cache
            cached_scores = []
            problems_to_evaluate = []
            
            for problem in problem_set:
                if problem in self.fitness_cache[agent.agent_id]:
                    cached_scores.append(self.fitness_cache[agent.agent_id][problem])
                else:
                    problems_to_evaluate.append(problem)
                    
            # If all problems are cached, return average
            if not problems_to_evaluate:
                return sum(cached_scores) / len(cached_scores)
                
            # Otherwise, evaluate remaining problems
            problem_set = problems_to_evaluate
            
        else:
            # Initialize cache for this agent
            self.fitness_cache[agent.agent_id] = {}
            cached_scores = []
            
        # For each problem, have agent reason through it
        scores = []
        
        for problem in problem_set:
            # Have agent reason through problem
            reasoning_result = agent.step_reasoning(problem, "")
            
            # Evaluate quality of reasoning (could be more sophisticated)
            reasoning_output = reasoning_result["reasoning_output"]
            
            # Simple heuristic: length and complexity
            length_score = min(len(reasoning_output) / 500, 1.0)  # Cap at 1.0
            complexity_score = len(set(reasoning_output.split())) / 200  # Vocabulary diversity
            
            # For a more sophisticated evaluation, we could use another LLM
            # to evaluate the quality of the reasoning
            
            # Combine scores
            score = (length_score + min(complexity_score, 1.0)) / 2
            
            # Add cognitive parameters influence
            score = score * (0.5 + agent.creativity * 0.25 + agent.attention_span * 0.25)
            
            # Ensure score is between 0 and 1
            score = max(0.0, min(1.0, score))
            
            # Cache the score
            self.fitness_cache[agent.agent_id][problem] = score
            
            scores.append(score)
            
        # Combine with cached scores
        all_scores = scores + cached_scores
        
        # Return average score
        return sum(all_scores) / len(all_scores)
        
    def _generate_problem_set(self, num_problems: int = 5) -> List[str]:
        """
        Generate a set of problems for fitness evaluation.
        
        Args:
            num_problems: Number of problems to generate
            
        Returns:
            List of problem statements
        """
        # Define problem templates for different domains
        problem_templates = [
            "How would you solve the problem of {issue} in {domain}?",
            "What approach would you take to address {issue} while considering {constraint}?",
            "Explain the relationship between {concept1} and {concept2} in the context of {domain}.",
            "Design a solution for {issue} that optimizes for both {goal1} and {goal2}.",
            "What are the implications of {development} for the future of {domain}?",
            "Compare and contrast {approach1} and {approach2} for solving {issue}.",
            "How might advances in {field} help address challenges in {domain}?",
            "What ethical considerations are important when applying {approach} to {issue}?"
        ]
        
        # Define variables for templates
        template_vars = {
            "issue": [
                "climate change", "resource scarcity", "income inequality", 
                "information overload", "algorithmic bias", "privacy concerns",
                "automation and job displacement", "mental health decline",
                "misinformation", "educational access", "healthcare costs",
                "artificial general intelligence risks", "quantum computing security threats"
            ],
            "domain": [
                "healthcare", "education", "transportation", "energy", 
                "agriculture", "finance", "governance", "media",
                "artificial intelligence", "environmental conservation", 
                "space exploration", "quantum computing", "neuroscience"
            ],
            "constraint": [
                "limited resources", "ethical considerations", "privacy requirements",
                "regulatory compliance", "scalability needs", "accessibility requirements",
                "environmental impact", "societal acceptance", "technical feasibility",
                "economic viability", "time constraints", "cross-cultural applicability"
            ],
            "concept1": [
                "emergence", "complexity", "causality", "consciousness", 
                "intelligence", "learning", "evolution", "cooperation",
                "competition", "adaptation", "resilience", "entropy",
                "information", "computation", "quantum effects"
            ],
            "concept2": [
                "systems thinking", "network effects", "feedback loops", 
                "self-organization", "optimization", "decision-making",
                "predictive modeling", "pattern recognition", "abstraction",
                "generalization", "creativity", "communication", "ethics"
            ],
            "goal1": [
                "efficiency", "sustainability", "equity", "innovation", 
                "resilience", "profitability", "security", "usability",
                "reliability", "transparency", "accountability", "scalability"
            ],
            "goal2": [
                "affordability", "accessibility", "privacy", "simplicity", 
                "environmental impact", "social benefit", "ethical integrity",
                "cultural sensitivity", "long-term viability", "adaptability"
            ],
            "development": [
                "artificial general intelligence", "quantum computing", 
                "blockchain technology", "CRISPR gene editing", "brain-computer interfaces",
                "renewable energy breakthroughs", "space colonization",
                "autonomous systems", "virtual/augmented reality", "synthetic biology"
            ],
            "approach1": [
                "centralized coordination", "market-based solutions", 
                "regulatory frameworks", "technological intervention",
                "community-based initiatives", "educational programs",
                "incentive structures", "open innovation", "systems thinking"
            ],
            "approach2": [
                "decentralized governance", "behavioral change", 
                "policy reform", "direct intervention", "research and development",
                "public-private partnerships", "international cooperation",
                "grassroots movements", "disruptive innovation"
            ],
            "field": [
                "artificial intelligence", "quantum computing", "materials science", 
                "renewable energy", "biotechnology", "neuroscience",
                "behavioral economics", "complex systems theory", "data science"
            ],
            "approach": [
                "machine learning", "genetic engineering", "surveillance technology", 
                "behavioral nudging", "predictive analytics", "automated decision systems",
                "blockchain solutions", "brain-computer interfaces", "social media algorithms"
            ]
        }
        
        # Generate problems
        problems = []
        for _ in range(num_problems):
            # Select a random template
            template = random.choice(problem_templates)
            
            # Fill in template variables
            problem = template
            for var_name in template_vars:
                if "{" + var_name + "}" in problem:
                    var_value = random.choice(template_vars[var_name])
                    problem = problem.replace("{" + var_name + "}", var_value)
                    
            problems.append(problem)
            
        return problems
        
    def _select_parents(self) -> List[Tuple[Agent, Agent]]:
        """
        Select parent pairs for reproduction based on fitness.
        
        Returns:
            List of parent pairs (tuples of agents)
        """
        # Number of offspring to create
        num_offspring = max(2, int(self.population_size * 0.5))
        
        # Get fitness scores
        agents_with_fitness = [(agent, self.fitness_scores.get(agent.agent_id, 0.5)) 
                              for agent in self.population]
        
        # Apply selection pressure
        fitness_values = np.array([fitness for _, fitness in agents_with_fitness])
        
        # Adjust fitness values with selection pressure
        if self.selection_pressure > 0:
            # Scale fitness values to emphasize differences
            adjusted_fitness = fitness_values ** (1.0 / self.selection_pressure)
        else:
            # Equal probability for all agents
            adjusted_fitness = np.ones_like(fitness_values)
            
        # Normalize to probability distribution
        selection_probs = adjusted_fitness / adjusted_fitness.sum()
        
        # Select parent pairs
        parent_pairs = []
        for _ in range(num_offspring):
            # Select two parents based on probabilities
            parents_idx = np.random.choice(
                len(self.population), 
                size=2, 
                replace=False, 
                p=selection_probs
            )
            
            parent1 = self.population[parents_idx[0]]
            parent2 = self.population[parents_idx[1]]
            
            parent_pairs.append((parent1, parent2))
            
        return parent_pairs
        
    def _create_offspring(self, parent_pairs: List[Tuple[Agent, Agent]]) -> List[Agent]:
        """
        Create offspring agents from parent pairs through crossover.
        
        Args:
            parent_pairs: List of parent pairs
            
        Returns:
            List of offspring agents
        """
        offspring = []
        
        for parent1, parent2 in parent_pairs:
            # Decide whether to perform crossover
            if random.random() < self.crossover_rate:
                # Create child through crossover
                child = self._crossover(parent1, parent2)
                offspring.append(child)
            else:
                # Clone from the more fit parent
                parent_to_clone = parent1 if (self.fitness_scores.get(parent1.agent_id, 0.5) > 
                                           self.fitness_scores.get(parent2.agent_id, 0.5)) else parent2
                                           
                child = self._clone_agent(parent_to_clone)
                offspring.append(child)
                
        return offspring
        
    def _crossover(self, parent1: Agent, parent2: Agent) -> Agent:
        """
        Create a new agent by combining traits from two parents.
        
        Args:
            parent1: First parent agent
            parent2: Second parent agent
            
        Returns:
            New agent with combined traits
        """
        # Combine names
        name_parts1 = parent1.name.split()
        name_parts2 = parent2.name.split()
        
        if len(name_parts1) > 1 and len(name_parts2) > 1:
            # Combine first and last names
            child_name = f"{name_parts1[0]} {name_parts2[-1]}"
        else:
            # If name structure is different, create new name
            child_name = f"Agent {uuid.uuid4().hex[:8]}"
            
        # Combine roles
        if random.random() < 0.5:
            child_role = parent1.role
        else:
            child_role = parent2.role
            
        # Combine backstories
        backstory_parts1 = parent1.backstory.split('. ')
        backstory_parts2 = parent2.backstory.split('. ')
        
        combined_backstory = []
        for i in range(min(len(backstory_parts1), len(backstory_parts2))):
            part = backstory_parts1[i] if random.random() < 0.5 else backstory_parts2[i]
            combined_backstory.append(part)
            
        child_backstory = '. '.join(combined_backstory) + '.'
        
        # Combine styles
        if random.random() < 0.5:
            child_style = parent1.style
        else:
            child_style = parent2.style
            
        # Combine instructions
        child_instructions = f"Combine the approaches of {parent1.role} and {parent2.role} to solve problems effectively."
        
        # Create child agent
        child = Agent(
            name=child_name,
            role=child_role,
            backstory=child_backstory,
            style=child_style,
            instructions=child_instructions,
            model=parent1.model,  # Use same model as parent
            db_manager=self.db_manager,
            global_workspace=parent1.global_workspace
        )
        
        # Combine cognitive parameters
        child.creativity = (parent1.creativity + parent2.creativity) / 2 + random.gauss(0, 0.1)
        child.attention_span = (parent1.attention_span + parent2.attention_span) / 2 + random.gauss(0, 0.1)
        child.learning_rate = (parent1.learning_rate + parent2.learning_rate) / 2 + random.gauss(0, 0.1)
        
        # Ensure parameters are within bounds
        child.creativity = max(0.0, min(1.0, child.creativity))
        child.attention_span = max(0.0, min(1.0, child.attention_span))
        child.learning_rate = max(0.0, min(1.0, child.learning_rate))
        
        # Initialize child's memory with selected memories from parents
        self._inherit_memories(child, parent1, parent2)
        
        return child
        
    def _mutate_offspring(self, offspring: List[Agent]) -> List[Agent]:
        """
        Apply mutations to offspring agents.
        
        Args:
            offspring: List of offspring agents
            
        Returns:
            List of mutated offspring agents
        """
        mutated_offspring = []
        
        for agent in offspring:
            # Check if mutation should occur
            if random.random() < self.mutation_rate:
                # Apply mutation
                mutated_agent = self._mutate_agent(agent)
                mutated_offspring.append(mutated_agent)
            else:
                # No mutation
                mutated_offspring.append(agent)
                
        return mutated_offspring
        
    def _mutate_agent(self, agent: Agent) -> Agent:
        """
        Apply mutation to a single agent.
        
        Args:
            agent: Agent to mutate
            
        Returns:
            Mutated agent
        """
        # Select mutation type
        mutation_type = random.choice([
            "cognitive_parameters",
            "role",
            "style",
            "instructions"
        ])
        
        if mutation_type == "cognitive_parameters":
            # Mutate cognitive parameters
            param = random.choice(["creativity", "attention_span", "learning_rate"])
            
            if param == "creativity":
                agent.creativity += random.gauss(0, 0.2)
                agent.creativity = max(0.0, min(1.0, agent.creativity))
            elif param == "attention_span":
                agent.attention_span += random.gauss(0, 0.2)
                agent.attention_span = max(0.0, min(1.0, agent.attention_span))
            elif param == "learning_rate":
                agent.learning_rate += random.gauss(0, 0.2)
                agent.learning_rate = max(0.0, min(1.0, agent.learning_rate))
                
            logger.debug(f"Mutated {agent.name}'s {param} to {getattr(agent, param)}")
            
        elif mutation_type == "role":
            # Enhance role with additional specialization
            specializations = [
                "with focus on interdisciplinary approaches",
                "specializing in complex systems",
                "with expertise in emergent phenomena",
                "focusing on practical applications",
                "with background in theoretical foundations",
                "emphasizing creative problem-solving",
                "with systems thinking approach"
            ]
            
            # Add specialization if role doesn't already have one
            if " with " not in agent.role and " specializing " not in agent.role:
                agent.role = f"{agent.role} {random.choice(specializations)}"
                logger.debug(f"Mutated {agent.name}'s role to {agent.role}")
                
        elif mutation_type == "style":
            # Modify communication style
            style_modifiers = [
                "with more analytical depth",
                "emphasizing clarity and precision",
                "with creative metaphors and analogies",
                "balancing big-picture thinking with details",
                "focusing on practical implications",
                "with socratic questioning approach",
                "emphasizing collaborative reasoning"
            ]
            
            agent.style = f"{agent.style} {random.choice(style_modifiers)}"
            logger.debug(f"Mutated {agent.name}'s style")
            
        elif mutation_type == "instructions":
            # Enhance instructions
            instruction_enhancements = [
                "Pay special attention to potential biases in reasoning.",
                "Consider multiple perspectives before drawing conclusions.",
                "Focus on finding innovative connections between different domains.",
                "Ensure that proposed solutions are practical and implementable.",
                "Balance creative thinking with analytical rigor.",
                "Explicitly identify assumptions in your reasoning process."
            ]
            
            agent.instructions = f"{agent.instructions} {random.choice(instruction_enhancements)}"
            logger.debug(f"Mutated {agent.name}'s instructions")
            
        return agent
        
    def _select_survivors(self, offspring: List[Agent]) -> None:
        """
        Select survivors for the next generation (elitism + offspring).
        
        Args:
            offspring: List of offspring agents
        """
        # Calculate how many elites to keep
        num_elites = max(1, int(self.population_size * 0.2))
        
        # Select elites based on fitness
        agents_with_fitness = [(agent, self.fitness_scores.get(agent.agent_id, 0.5)) 
                              for agent in self.population]
        agents_with_fitness.sort(key=lambda x: x[1], reverse=True)
        
        elites = [agent for agent, _ in agents_with_fitness[:num_elites]]
        
        # Build new population from elites and offspring
        new_population = elites[:]
        
        # Add offspring, prioritizing diversity
        for agent in offspring:
            # Skip if population is full
            if len(new_population) >= self.population_size:
                break
                
            # Add agent to population
            new_population.append(agent)
            
        # If we still need more agents, create random ones
        remaining = self.population_size - len(new_population)
        if remaining > 0:
            domains = ["science", "humanities", "engineering", "arts", "business"]
            
            for i in range(remaining):
                domain = domains[i % len(domains)]
                agent = self.agent_spawner.spawn_agent_by_type(domain)
                
                if agent:
                    new_population.append(agent)
                    
        # Update population
        self.population = new_population
        
        # Reset fitness scores for new agents
        for agent in self.population:
            if agent.agent_id not in self.fitness_scores:
                self.fitness_scores[agent.agent_id] = 0.5  # Default score
                
        logger.info(f"Selected {len(self.population)} survivors: {num_elites} elites + {len(offspring)} offspring")
        
    def _clone_agent(self, parent: Agent) -> Agent:
        """
        Create a clone of an agent with minor variations.
        
        Args:
            parent: Parent agent to clone
            
        Returns:
            Cloned agent
        """
        # Create clone with slight name variation
        name_parts = parent.name.split()
        if len(name_parts) > 1:
            clone_name = f"{name_parts[0]} {name_parts[-1]}-{uuid.uuid4().hex[:4]}"
        else:
            clone_name = f"{parent.name}-{uuid.uuid4().hex[:4]}"
            
        # Create clone
        clone = Agent(
            name=clone_name,
            role=parent.role,
            backstory=parent.backstory,
            style=parent.style,
            instructions=parent.instructions,
            model=parent.model,
            db_manager=self.db_manager,
            global_workspace=parent.global_workspace
        )
        
        # Copy cognitive parameters with small random variations
        clone.creativity = parent.creativity + random.gauss(0, 0.05)
        clone.attention_span = parent.attention_span + random.gauss(0, 0.05)
        clone.learning_rate = parent.learning_rate + random.gauss(0, 0.05)
        
        # Ensure parameters are within bounds
        clone.creativity = max(0.0, min(1.0, clone.creativity))
        clone.attention_span = max(0.0, min(1.0, clone.attention_span))
        clone.learning_rate = max(0.0, min(1.0, clone.learning_rate))
        
        # Inherit memories
        self._inherit_memories(clone, parent, None)
        
        return clone
        
    def _inherit_memories(self, child: Agent, parent1: Agent, parent2: Optional[Agent] = None) -> None:
        """
        Have a child agent inherit memories from parents.
        
        Args:
            child: Child agent
            parent1: First parent agent
            parent2: Optional second parent agent
        """
        # Sample memories from parent1
        ltm1 = parent1.memory_module.long_term_memory
        semantic1 = parent1.memory_module.semantic_memory
        
        # If parent2 is available, also sample from parent2
        if parent2:
            ltm2 = parent2.memory_module.long_term_memory
            semantic2 = parent2.memory_module.semantic_memory
            
            # Combine LTM memories
            all_ltm = ltm1 + ltm2
            
            # Combine semantic memories
            all_semantic = {}
            all_semantic.update(semantic1)
            all_semantic.update(semantic2)
        else:
            all_ltm = ltm1
            all_semantic = semantic1
            
        # Sample from LTM (long-term memory)
        if all_ltm:
            num_to_inherit = min(len(all_ltm), 10)  # Inherit up to 10 memories
            inherited_ltm = random.sample(all_ltm, num_to_inherit)
            
            # Add to child's long-term memory
            for memory in inherited_ltm:
                # Create a copy with new ID
                memory_copy = memory.copy()
                memory_copy['id'] = f"inherited_{uuid.uuid4().hex}"
                memory_copy['source'] = f"inherited_from_{parent1.name}"
                memory_copy['importance'] *= 0.9  # Slightly reduce importance
                
                child.memory_module.long_term_memory.append(memory_copy)
                
        # Sample from semantic memory
        if all_semantic:
            num_to_inherit = min(len(all_semantic), 5)  # Inherit up to 5 semantic memories
            semantic_keys = random.sample(list(all_semantic.keys()), num_to_inherit)
            
            # Add to child's semantic memory
            for key in semantic_keys:
                memory = all_semantic[key]
                
                # Create a copy with new ID
                memory_copy = memory.copy()
                memory_copy['id'] = f"inherited_{uuid.uuid4().hex}"
                memory_copy['source'] = f"inherited_from_{parent1.name if key in semantic1 else parent2.name}"
                memory_copy['importance'] *= 0.9  # Slightly reduce importance
                
                child.memory_module.semantic_memory[key] = memory_copy
                
    def _record_generation(self) -> None:
        """Record data about the current generation for analysis."""
        generation_data = {
            "generation": self.generation,
            "timestamp": datetime.datetime.now().isoformat(),
            "population_size": len(self.population),
            "agents": [{
                "id": agent.agent_id,
                "name": agent.name,
                "role": agent.role,
                "fitness": self.fitness_scores.get(agent.agent_id, 0.5),
                "cognitive_parameters": {
                    "creativity": agent.creativity,
                    "attention_span": agent.attention_span,
                    "learning_rate": agent.learning_rate
                }
            } for agent in self.population],
            "average_fitness": sum(self.fitness_scores.values()) / max(1, len(self.fitness_scores)),
            "max_fitness": max(self.fitness_scores.values()) if self.fitness_scores else 0,
            "parameters": {
                "mutation_rate": self.mutation_rate,
                "crossover_rate": self.crossover_rate,
                "selection_pressure": self.selection_pressure
            }
        }
        
        # Add to history
        self.evolution_history.append(generation_data)
        
        # Save to file
        filename = os.path.join(self.evolution_data_dir, f"generation_{self.generation}.json")
        with open(filename, 'w') as f:
            json.dump(generation_data, f, indent=2)
            
        logger.debug(f"Recorded generation {self.generation} data")
        
    def save_evolution_state(self, filename: str) -> None:
        """
        Save evolution state to file.
        
        Args:
            filename: Filename to save to
        """
        # Create state dictionary
        state = {
            "generation": self.generation,
            "fitness_scores": self.fitness_scores,
            "evolution_history": self.evolution_history,
            "parameters": {
                "mutation_rate": self.mutation_rate,
                "crossover_rate": self.crossover_rate,
                "selection_pressure": self.selection_pressure,
                "population_size": self.population_size
            },
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"Saved evolution state to {filename}")
        
        # Also save agent states
        agent_dir = filename.replace('.json', '_agents')
        os.makedirs(agent_dir, exist_ok=True)
        
        for i, agent in enumerate(self.population):
            agent_filename = os.path.join(agent_dir, f"agent_{i}.json")
            agent.save_agent_state(agent_filename)
            
    def load_evolution_state(self, filename: str) -> None:
        """
        Load evolution state from file.
        
        Args:
            filename: Filename to load from
        """
        # Load from file
        with open(filename, 'r') as f:
            state = json.load(f)
            
        # Restore state
        self.generation = state["generation"]
        self.fitness_scores = state["fitness_scores"]
        self.evolution_history = state["evolution_history"]
        
        # Restore parameters
        params = state["parameters"]
        self.mutation_rate = params["mutation_rate"]
        self.crossover_rate = params["crossover_rate"]
        self.selection_pressure = params["selection_pressure"]
        self.population_size = params["population_size"]
        
        logger.info(f"Loaded evolution state from {filename}")
        
        # Also load agent states if available
        agent_dir = filename.replace('.json', '_agents')
        if os.path.exists(agent_dir):
            self.population = []
            
            # Find all agent files
            agent_files = [f for f in os.listdir(agent_dir) if f.endswith('.json')]
            
            for agent_file in agent_files:
                # Create agent
                agent = Agent(
                    name="Temporary",
                    role="Temporary",
                    backstory="Temporary",
                    style="Temporary",
                    instructions="Temporary",
                    model=self.agent_spawner.model,
                    db_manager=self.db_manager,
                    global_workspace=None  # Will be set after loading
                )
                
                # Load agent state
                agent_filename = os.path.join(agent_dir, agent_file)
                agent.load_agent_state(agent_filename)
                
                # Add to population
                self.population.append(agent)
                
            logger.info(f"Loaded {len(self.population)} agents from {agent_dir}")

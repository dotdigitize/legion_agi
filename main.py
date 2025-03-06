"""
Legion AGI System - Main Execution Script

This script provides the main entry point for the Legion AGI system, 
supporting both tool mode and continuous evolution mode.
"""

import argparse
import os
import time
import datetime
import signal
import sys
import uuid
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

import ollama

# Import Legion AGI components
from legion_agi.config import (
    LOG_FILE, LOG_LEVEL, LOG_FORMAT, LOG_ROTATION,
    DEFAULT_MODEL, TOOL_MODE, EVOLUTION_MODE,
    GENERATIONS_PER_CYCLE
)
from legion_agi.core.global_workspace import GlobalWorkspace
from legion_agi.utils.db_manager import DatabaseManager
from legion_agi.agents.spawning import AgentSpawner
from legion_agi.agents.evolution import AgentEvolution
from legion_agi.methods.past import PASTMethod
from legion_agi.methods.raft import RAFTMethod
from legion_agi.methods.eat import EATMethod
from legion_agi.utils.visualization import SystemVisualizer


# Configure logging
logger.remove()  # Remove default handler
logger.add(sys.stderr, level=LOG_LEVEL)
logger.add(LOG_FILE, rotation=LOG_ROTATION, level=LOG_LEVEL, format=LOG_FORMAT)


class LegionAGI:
    """
    Main Legion AGI system class.
    Coordinates all components and provides the primary interface.
    """
    
    def __init__(
        self, 
        model: str = DEFAULT_MODEL,
        mode: str = TOOL_MODE,
        session_id: Optional[str] = None,
        data_dir: str = "data",
        visualize: bool = True
    ):
        """
        Initialize Legion AGI system.
        
        Args:
            model: LLM model to use
            mode: Operation mode (tool or evolve)
            session_id: Session ID for persistence
            data_dir: Directory for data storage
            visualize: Whether to enable visualizations
        """
        # Basic system parameters
        self.model = model
        self.mode = mode
        self.session_id = session_id or str(uuid.uuid4())
        self.data_dir = data_dir
        self.visualize = visualize
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Save data in session-specific directory
        self.session_dir = os.path.join(data_dir, f"session_{self.session_id}")
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Initialize visualization if enabled
        self.visualizer = SystemVisualizer(os.path.join(self.session_dir, "viz")) if visualize else None
        
        # Initialize system components
        logger.info(f"Initializing Legion AGI system with mode: {mode}")
        
        # Database for persistence
        db_path = os.path.join(self.session_dir, "legion_agi.db")
        self.db_manager = DatabaseManager(self.session_id, db_path)
        
        # Global workspace for information integration
        self.global_workspace = GlobalWorkspace()
        
        # Agent spawner for creating specialized agents
        self.agent_spawner = AgentSpawner(
            model=self.model,
            db_manager=self.db_manager,
            global_workspace=self.global_workspace
        )
        
        # Reasoning methods
        self.past_method = None  # Will be initialized with agents
        self.raft_method = None  # Will be initialized with agents
        self.eat_method = None   # Will be initialized with agents
        
        # For evolution mode
        self.agent_evolution = None
        if self.mode == EVOLUTION_MODE:
            self.agent_evolution = AgentEvolution(
                agent_spawner=self.agent_spawner,
                db_manager=self.db_manager,
                evolution_data_dir=os.path.join(self.session_dir, "evolution")
            )
            
        # System state
        self.agents = []
        self.conversation_history = []
        self.original_question = None
        self.current_solution = None
        self.running = False
        self.evolution_generation = 0
        
        logger.info(f"Legion AGI system initialized with session ID: {self.session_id}")
        
    def start(self) -> None:
        """Start the Legion AGI system."""
        self.running = True
        
        if self.mode == EVOLUTION_MODE:
            logger.info("Starting Legion AGI in evolution mode")
            self._run_evolution_mode()
        else:
            logger.info("Starting Legion AGI in tool mode")
            # Tool mode is interactive, controlled by process_query
            
    def stop(self) -> None:
        """Stop the Legion AGI system."""
        logger.info("Stopping Legion AGI system")
        self.running = False
        
        # Save state before stopping
        self.save_state()
        
        # Close database connection
        self.db_manager.close()
        
    def save_state(self) -> str:
        """
        Save the current state of the system.
        
        Returns:
            Path to the saved state file
        """
        # Create state directory if needed
        state_dir = os.path.join(self.session_dir, "state")
        os.makedirs(state_dir, exist_ok=True)
        
        # Create timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        state_file = os.path.join(state_dir, f"system_state_{timestamp}.json")
        
        # Simple state saving for now
        import json
        state = {
            "session_id": self.session_id,
            "mode": self.mode,
            "model": self.model,
            "timestamp": timestamp,
            "conversation_history": self.conversation_history,
            "original_question": self.original_question,
            "evolution_generation": self.evolution_generation if self.mode == EVOLUTION_MODE else 0
        }
        
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"System state saved to {state_file}")
        
        # If in evolution mode, also save evolution state
        if self.mode == EVOLUTION_MODE and self.agent_evolution:
            evolution_state_file = os.path.join(state_dir, f"evolution_state_{timestamp}.json")
            self.agent_evolution.save_evolution_state(evolution_state_file)
            
        return state_file
        
    def load_state(self, state_file: str) -> bool:
        """
        Load system state from file.
        
        Args:
            state_file: Path to the state file
            
        Returns:
            Success status
        """
        try:
            import json
            with open(state_file, "r") as f:
                state = json.load(f)
                
            # Restore basic state
            self.session_id = state.get("session_id", self.session_id)
            self.mode = state.get("mode", self.mode)
            self.model = state.get("model", self.model)
            self.conversation_history = state.get("conversation_history", [])
            self.original_question = state.get("original_question", None)
            self.evolution_generation = state.get("evolution_generation", 0)
            
            logger.info(f"System state loaded from {state_file}")
            
            # If in evolution mode, check for evolution state
            if self.mode == EVOLUTION_MODE and self.agent_evolution:
                evolution_state_file = state_file.replace("system_state", "evolution_state")
                if os.path.exists(evolution_state_file):
                    self.agent_evolution.load_evolution_state(evolution_state_file)
                    
            return True
            
        except Exception as e:
            logger.error(f"Error loading state from {state_file}: {e}")
            return False
            
    def process_query(self, query: str, method: str = "past") -> Dict[str, Any]:
        """
        Process a user query using the specified reasoning method.
        This is the main entry point for tool mode.
        
        Args:
            query: User query/question
            method: Reasoning method to use (past, raft, eat)
            
        Returns:
            Result dictionary with the processed solution
        """
        logger.info(f"Processing query using {method.upper()} method: {query[:100]}...")
        
        # Store original question
        self.original_question = query
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "User",
            "content": query,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Spawn agents based on the query
        self.agents = self.agent_spawner.analyze_input_and_spawn_agents(query)
        
        if not self.agents:
            error_msg = "Failed to spawn agents for this query."
            logger.error(error_msg)
            return {"error": error_msg}
            
        # Visualize agent network if enabled
        if self.visualizer:
            self.visualizer.visualize_agent_network(self.agents)
            
        # Initialize reasoning methods with current agents
        self.past_method = PASTMethod(
            agents=self.agents,
            global_workspace=self.global_workspace,
            db_manager=self.db_manager,
            model=self.model
        )
        
        self.raft_method = RAFTMethod(
            agents=self.agents,
            global_workspace=self.global_workspace,
            db_manager=self.db_manager,
            model=self.model
        )
        
        self.eat_method = EATMethod(
            agents=self.agents,
            global_workspace=self.global_workspace,
            db_manager=self.db_manager,
            model=self.model
        )
        
        # Process with the selected method
        if method.lower() == "past":
            solution = self.past_method.execute_full_method(query)
        elif method.lower() == "raft":
            solution = self.raft_method.execute_full_method(query)
        elif method.lower() == "eat":
            solution = self.eat_method.execute_full_method(query)
        else:
            # Default to PAST method
            solution = self.past_method.execute_full_method(query)
            
        # Store result
        self.current_solution = solution
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "System",
            "content": solution.get("integrated_solution", "No solution generated."),
            "timestamp": datetime.datetime.now().isoformat(),
            "method": method
        })
        
        # Visualize if enabled
        if self.visualizer:
            # Visualize agent interactions
            self.visualizer.visualize_agent_interaction(self.conversation_history)
            
            # Visualize global workspace
            self.visualizer.visualize_global_workspace(self.global_workspace)
            
            # Visualize memory of a representative agent
            if self.agents:
                self.visualizer.visualize_memory_state(self.agents[0].memory_module)
                
        # Save state
        self.save_state()
        
        return solution
        
    def _run_evolution_mode(self) -> None:
        """Run the system in continuous evolution mode."""
        if not self.agent_evolution:
            logger.error("Agent evolution not initialized for evolution mode")
            return
            
        logger.info("Running Legion AGI in continuous evolution mode")
        
        try:
            # Initialize agent population
            logger.info("Initializing agent population")
            self.agent_evolution.initialize_population()
            self.agents = self.agent_evolution.population
            
            # Main evolution loop
            while self.running:
                self.evolution_generation += 1
                logger.info(f"Starting evolution generation {self.evolution_generation}")
                
                # Generate problem set for this generation
                problem_set = self._generate_evolution_problems()
                
                # Evaluate agents on the problem set
                self.agent_evolution.evaluate_population_fitness(problem_set)
                
                # Visualize current state
                if self.visualizer:
                    # Visualize agent network
                    self.visualizer.visualize_agent_network(
                        self.agents,
                        filename=f"agent_network_gen_{self.evolution_generation}.png"
                    )
                    
                    # Visualize global workspace
                    self.visualizer.visualize_global_workspace(
                        self.global_workspace,
                        filename=f"global_workspace_gen_{self.evolution_generation}.png"
                    )
                    
                # Evolve population
                logger.info(f"Evolving population for generation {self.evolution_generation}")
                self.agents = self.agent_evolution.evolve_population(GENERATIONS_PER_CYCLE)
                
                # Save state
                self.save_state()
                
                # Sleep between generations to avoid resource exhaustion
                time.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("Evolution mode interrupted by user")
            self.stop()
        except Exception as e:
            logger.error(f"Error in evolution mode: {e}")
            self.stop()
            
    def _generate_evolution_problems(self) -> List[str]:
        """
        Generate problems for evolution evaluation.
        
        Returns:
            List of problem statements
        """
        # For demonstration, use a mix of fixed problems and variations
        base_problems = [
            "Explain the relationship between quantum mechanics and consciousness.",
            "Design a sustainable urban transportation system for a city of 1 million people.",
            "Develop a theoretical framework for artificial general intelligence.",
            "Propose a solution to the Fermi paradox.",
            "Design an educational curriculum for teaching complex systems thinking."
        ]
        
        # Add variations based on generation number
        variations = [
            f"How would {base_problems[i % len(base_problems)].lower()} be different in {50 + self.evolution_generation} years?"
            for i in range(3)
        ]
        
        # Combine problems
        return base_problems + variations
        
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current status of the system.
        
        Returns:
            Status dictionary
        """
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "model": self.model,
            "running": self.running,
            "num_agents": len(self.agents),
            "conversation_length": len(self.conversation_history),
            "evolution_generation": self.evolution_generation if self.mode == EVOLUTION_MODE else 0,
            "agent_types": [agent.role for agent in self.agents],
            "original_question": self.original_question,
            "timestamp": datetime.datetime.now().isoformat()
        }


def main():
    """Main entry point for Legion AGI system."""
    parser = argparse.ArgumentParser(description="Legion AGI System")
    parser.add_argument("--mode", type=str, default=TOOL_MODE, 
                       choices=[TOOL_MODE, EVOLUTION_MODE],
                       help="Operation mode (tool or evolve)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                       help="LLM model to use")
    parser.add_argument("--session", type=str, default=None,
                       help="Session ID for persistence")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Directory for data storage")
    parser.add_argument("--load-state", type=str, default=None,
                       help="Load state from file")
    parser.add_argument("--query", type=str, default=None,
                       help="Query to process (tool mode only)")
    parser.add_argument("--method", type=str, default="past",
                       choices=["past", "raft", "eat"],
                       help="Reasoning method to use (tool mode only)")
    parser.add_argument("--no-visualize", action="store_true",
                       help="Disable visualizations")
    
    args = parser.parse_args()
    
    # Create system
    legion = LegionAGI(
        model=args.model,
        mode=args.mode,
        session_id=args.session,
        data_dir=args.data_dir,
        visualize=not args.no_visualize
    )
    
    # Load state if specified
    if args.load_state:
        legion.load_state(args.load_state)
        
    # Register signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutting down Legion AGI system...")
        legion.stop()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    # Process query if provided (tool mode)
    if args.query and args.mode == TOOL_MODE:
        result = legion.process_query(args.query, args.method)
        print("\nResult:")
        if "integrated_solution" in result:
            print(result["integrated_solution"])
        else:
            print(result)
        legion.stop()
    # Otherwise, start interactive mode
    else:
        legion.start()
        
        # If in tool mode, run interactive loop
        if args.mode == TOOL_MODE:
            try:
                print("\nLegion AGI System - Interactive Mode")
                print("Enter '/exit' to quit, '/status' to view system status")
                print("Enter your question or command:")
                
                while legion.running:
                    query = input("\nYou: ").strip()
                    
                    if query.lower() in ['/exit', '/quit']:
                        legion.stop()
                        break
                    elif query.lower() == '/status':
                        status = legion.get_system_status()
                        print("\nSystem Status:")
                        for key, value in status.items():
                            print(f"  {key}: {value}")
                    elif query.lower().startswith('/method '):
                        method = query.split()[1].lower()
                        if method in ["past", "raft", "eat"]:
                            print(f"Reasoning method set to: {method.upper()}")
                        else:
                            print(f"Unknown method: {method}")
                    elif query.lower().startswith('/eval '):
                        # TODO: Implement command to evaluate specific agent
                        print("Evaluation not yet implemented")
                    elif query.lower().startswith('/save'):
                        state_file = legion.save_state()
                        print(f"State saved to: {state_file}")
                    elif query.lower().startswith('/help'):
                        print("\nAvailable commands:")
                        print("  /exit, /quit - Exit the system")
                        print("  /status - Show system status")
                        print("  /method [past|raft|eat] - Set reasoning method")
                        print("  /save - Save current system state")
                        print("  /help - Show this help message")
                    elif query:
                        print("\nProcessing your query...")
                        result = legion.process_query(query, args.method)
                        print("\nLegion AGI:")
                        if "integrated_solution" in result:
                            print(result["integrated_solution"])
                        else:
                            print(result)
                            
            except KeyboardInterrupt:
                print("\nShutting down Legion AGI system...")
                legion.stop()
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                legion.stop()


if __name__ == "__main__":
    main()
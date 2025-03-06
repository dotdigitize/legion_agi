"""
Command Line Interface for Legion AGI

This module provides a comprehensive command-line interface for interacting with
the Legion AGI system, supporting both interactive and batch processing modes.
Features include command history, auto-completion, and formatted output.
"""

import os
import sys
import cmd
import json
import argparse
import time
import readline
import threading
import subprocess
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime
import shutil

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.text import Text
from rich.tree import Tree
from rich.traceback import install as install_rich_traceback

from loguru import logger

from legion_agi.main import LegionAGI
from legion_agi.config import (
    DEFAULT_MODEL,
    TOOL_MODE,
    EVOLUTION_MODE,
    DATA_DIR,
    LOG_DIR
)
from legion_agi.utils.visualization import SystemVisualizer


# Install rich traceback handler
install_rich_traceback()

# Terminal dimensions
terminal_width, terminal_height = shutil.get_terminal_size()

# Initialize rich console
console = Console()


class LegionCLI(cmd.Cmd):
    """
    Interactive command-line interface for Legion AGI.
    Provides command history, auto-completion, and rich formatted output.
    """
    
    intro = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║                        LEGION AGI SYSTEM                         ║
║                                                                  ║
║  Type 'help' or '?' to list commands.                           ║
║  Type 'start' to initialize the system.                         ║
║  Type 'exit' to quit.                                           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    prompt = "[Legion AGI] > "
    
    def __init__(self, 
                model: str = DEFAULT_MODEL,
                mode: str = TOOL_MODE,
                session_id: Optional[str] = None,
                data_dir: str = DATA_DIR,
                visualize: bool = True):
        """
        Initialize Legion CLI.
        
        Args:
            model: LLM model to use
            mode: Operation mode
            session_id: Optional session ID
            data_dir: Data directory
            visualize: Enable visualizations
        """
        super().__init__()
        self.model = model
        self.mode = mode
        self.session_id = session_id
        self.data_dir = data_dir
        self.visualize = visualize
        
        # Legion AGI system (initialized with start command)
        self.legion = None
        
        # CLI state
        self.running = False
        self.processing = False
        self.current_query = None
        self.last_result = None
        self.evolution_thread = None
        
        # Command history
        self.command_history = []
        self.history_file = os.path.join(data_dir, ".legion_history")
        
        # Initialize readline for history and auto-completion
        self._setup_readline()
        
    def _setup_readline(self) -> None:
        """Setup readline configuration for command history and completion."""
        # Load command history if exists
        if os.path.exists(self.history_file):
            try:
                readline.read_history_file(self.history_file)
                readline.set_history_length(1000)
            except Exception as e:
                logger.warning(f"Could not read history file: {e}")
                
        # Set completer and delimiters
        readline.set_completer(self.complete)
        readline.parse_and_bind("tab: complete")
        
    def preloop(self) -> None:
        """Actions before entering the command loop."""
        # Display intro
        console.print(Panel.fit(self.intro.strip(), border_style="blue"))
        
    def postloop(self) -> None:
        """Actions after exiting the command loop."""
        # Save command history
        try:
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
            readline.write_history_file(self.history_file)
        except Exception as e:
            logger.warning(f"Could not write history file: {e}")
            
        # Stop Legion AGI if running
        if self.legion and self.legion.running:
            self.legion.stop()
            
        console.print(Panel("Legion AGI system shutdown complete.", border_style="yellow"))
        
    def emptyline(self) -> bool:
        """Handle empty line input (do nothing)."""
        return False
        
    def default(self, line: str) -> bool:
        """
        Handle unknown commands.
        Treats them as queries if the system is running.
        """
        if not line.strip():
            return False
            
        if self.legion and self.legion.running:
            return self.do_query(line)
        else:
            console.print(f"Unknown command: {line}", style="red")
            console.print("Type 'help' for a list of commands.")
            return False
            
    def do_exit(self, arg: str) -> bool:
        """Exit the CLI."""
        console.print("Shutting down Legion AGI system...", style="yellow")
        return True
        
    def do_quit(self, arg: str) -> bool:
        """Alias for 'exit'."""
        return self.do_exit(arg)
        
    def do_EOF(self, arg: str) -> bool:
        """Handle Ctrl+D (EOF)."""
        console.print("\nReceived EOF (Ctrl+D). Exiting...", style="yellow")
        return True
        
    def do_start(self, arg: str) -> None:
        """
        Start the Legion AGI system.
        Usage: start [--load STATE_FILE]
        """
        if self.legion and self.legion.running:
            console.print("Legion AGI system is already running.", style="yellow")
            return
            
        console.print("Initializing Legion AGI system...", style="blue")
        
        # Parse additional arguments
        args = arg.split()
        load_state = None
        
        if "--load" in args and args.index("--load") + 1 < len(args):
            load_idx = args.index("--load")
            load_state = args[load_idx + 1]
            
        # Initialize progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading...", total=None)
            
            # Initialize Legion AGI
            self.legion = LegionAGI(
                model=self.model,
                mode=self.mode,
                session_id=self.session_id,
                data_dir=self.data_dir,
                visualize=self.visualize
            )
            
            # Load state if specified
            if load_state:
                progress.update(task, description=f"Loading state from {load_state}...")
                self.legion.load_state(load_state)
                
            # Start system
            progress.update(task, description="Starting Legion AGI system...")
            self.legion.running = True
            self.running = True
            
            # Start evolution thread if in evolution mode
            if self.mode == EVOLUTION_MODE:
                progress.update(task, description="Starting evolution process...")
                self.evolution_thread = threading.Thread(
                    target=self._run_evolution_thread,
                    daemon=True
                )
                self.evolution_thread.start()
                
        # Display system status
        status = self.legion.get_system_status()
        self._display_status(status)
        
        console.print("Legion AGI system is now running.", style="green")
        console.print("Type 'query <your question>' or simply type your question to begin.")
        
    def do_status(self, arg: str) -> None:
        """Display system status."""
        if not self.legion:
            console.print("Legion AGI system is not initialized. Use 'start' command.", style="yellow")
            return
            
        status = self.legion.get_system_status()
        self._display_status(status)
        
    def do_query(self, arg: str) -> None:
        """
        Process a query with Legion AGI.
        Usage: query <your question>
        """
        if not self.legion or not self.legion.running:
            console.print("Legion AGI system is not running. Use 'start' command.", style="yellow")
            return
            
        if not arg.strip():
            console.print("Please provide a query.", style="yellow")
            return
            
        # Cancel if already processing
        if self.processing:
            console.print("Already processing a query. Please wait or press Ctrl+C to cancel.", style="yellow")
            return
            
        self.processing = True
        self.current_query = arg
        
        try:
            # Determine reasoning method
            method = "past"  # Default method
            if arg.startswith("/past "):
                method = "past"
                arg = arg[6:].strip()
            elif arg.startswith("/raft "):
                method = "raft"
                arg = arg[6:].strip()
            elif arg.startswith("/eat "):
                method = "eat"
                arg = arg[5:].strip()
                
            # Display thinking animation
            console.print(f"\nProcessing query using {method.upper()} method:", style="blue")
            console.print(Panel(arg, style="cyan"))
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                # Add task for tracking progress
                task = progress.add_task("Thinking...", total=None)
                
                # Process query in a separate thread to keep UI responsive
                result_container = [None]
                
                def process_query():
                    try:
                        result = self.legion.process_query(arg, method)
                        result_container[0] = result
                    except Exception as e:
                        logger.exception(f"Error processing query: {e}")
                        result_container[0] = {"error": str(e)}
                        
                # Start processing thread
                thread = threading.Thread(target=process_query)
                thread.start()
                
                # Update progress while waiting
                thinking_states = [
                    "Analyzing question...",
                    "Spawning specialized agents...",
                    "Agents are collaborating...",
                    "Refining solutions...",
                    "Integrating expert knowledge...",
                    "Evaluating approach...",
                    "Finalizing response..."
                ]
                
                i = 0
                while thread.is_alive():
                    progress.update(task, description=thinking_states[i % len(thinking_states)])
                    i += 1
                    time.sleep(1.0)
                    
                # Wait for thread to complete
                thread.join()
                
                # Get result
                result = result_container[0]
                self.last_result = result
                
            # Display result
            if result:
                self._display_result(result)
            else:
                console.print("No result returned.", style="red")
                
        except KeyboardInterrupt:
            console.print("\nQuery processing interrupted.", style="yellow")
        except Exception as e:
            logger.exception(f"Error in query processing: {e}")
            console.print(f"\nError processing query: {e}", style="red")
        finally:
            self.processing = False
            
    def do_method(self, arg: str) -> None:
        """
        Set the default reasoning method.
        Usage: method [past|raft|eat]
        """
        if not arg.strip():
            console.print("Current methods: past, raft, eat", style="blue")
            console.print("Usage examples:", style="blue")
            console.print("  method past   - Set default method to PAST", style="blue")
            console.print("  /past <query> - Use PAST method for a specific query", style="blue")
            return
            
        method = arg.lower().strip()
        if method in ["past", "raft", "eat"]:
            console.print(f"Default reasoning method set to: {method.upper()}", style="green")
        else:
            console.print(f"Unknown method: {method}", style="red")
            console.print("Available methods: past, raft, eat", style="yellow")
            
    def do_save(self, arg: str) -> None:
        """
        Save current system state.
        Usage: save [filename]
        """
        if not self.legion:
            console.print("Legion AGI system is not initialized. Use 'start' command.", style="yellow")
            return
            
        try:
            # Get filename if provided
            filename = arg.strip() if arg.strip() else None
            
            # Save state
            state_file = self.legion.save_state()
            
            console.print(f"State saved to: {state_file}", style="green")
            
        except Exception as e:
            logger.exception(f"Error saving state: {e}")
            console.print(f"Error saving state: {e}", style="red")
            
    def do_agents(self, arg: str) -> None:
        """Display information about the current agents."""
        if not self.legion or not hasattr(self.legion, 'agents') or not self.legion.agents:
            console.print("No agents available. Start the system and process a query first.", style="yellow")
            return
            
        # Create table of agents
        table = Table(title="Current Agents")
        table.add_column("Name", style="cyan")
        table.add_column("Role", style="green")
        table.add_column("Cognitive Parameters", style="magenta")
        
        for agent in self.legion.agents:
            cognitive_params = f"C:{agent.creativity:.2f} A:{agent.attention_span:.2f} L:{agent.learning_rate:.2f}"
            table.add_row(agent.name, agent.role, cognitive_params)
            
        console.print(table)
        
        # Display visualization if available
        if self.visualizer:
            console.print(f"Agent network visualization available at: {self.legion.session_dir}/viz/", style="blue")
            
    def do_viz(self, arg: str) -> None:
        """
        Open visualization files.
        Usage: viz [type]
        Types: agents, memory, workspace, reasoning, all
        """
        if not self.legion or not self.legion.visualize:
            console.print("Visualizations are not enabled.", style="yellow")
            return
            
        viz_dir = os.path.join(self.legion.session_dir, "viz")
        if not os.path.exists(viz_dir):
            console.print("No visualizations available yet.", style="yellow")
            return
            
        # Determine visualization type
        viz_type = arg.lower().strip() if arg.strip() else "all"
        
        # Get all visualization files
        files = os.listdir(viz_dir)
        
        # Filter based on type
        if viz_type == "agents":
            files = [f for f in files if "agent" in f.lower()]
        elif viz_type == "memory":
            files = [f for f in files if "memory" in f.lower()]
        elif viz_type == "workspace":
            files = [f for f in files if "workspace" in f.lower()]
        elif viz_type == "reasoning":
            files = [f for f in files if "reasoning" in f.lower()]
            
        if not files:
            console.print(f"No {viz_type} visualizations found.", style="yellow")
            return
            
        # Display available visualizations
        table = Table(title=f"{viz_type.capitalize()} Visualizations")
        table.add_column("Filename", style="cyan")
        table.add_column("Size", style="green")
        table.add_column("Created", style="magenta")
        
        for file in files:
            file_path = os.path.join(viz_dir, file)
            size = os.path.getsize(file_path)
            created = datetime.fromtimestamp(os.path.getctime(file_path))
            
            # Format size
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size / (1024 * 1024):.1f} MB"
                
            table.add_row(file, size_str, created.strftime("%Y-%m-%d %H:%M:%S"))
            
        console.print(table)
        
        # Open latest visualization if few files
        if len(files) <= 3:
            try:
                latest_file = max(files, key=lambda f: os.path.getctime(os.path.join(viz_dir, f)))
                latest_path = os.path.join(viz_dir, latest_file)
                
                console.print(f"Opening latest visualization: {latest_file}", style="blue")
                
                # Try to open with platform-specific command
                if sys.platform.startswith('darwin'):  # macOS
                    subprocess.Popen(['open', latest_path])
                elif sys.platform.startswith('win'):   # Windows
                    os.startfile(latest_path)
                else:  # Linux and others
                    subprocess.Popen(['xdg-open', latest_path])
                    
            except Exception as e:
                console.print(f"Could not open visualization: {e}", style="red")
                
    def do_logs(self, arg: str) -> None:
        """
        Display recent log entries.
        Usage: logs [num_lines] [--level LEVEL]
        """
        # Parse arguments
        args = arg.split()
        num_lines = 20  # Default
        level = None
        
        for i, a in enumerate(args):
            if a.isdigit():
                num_lines = int(a)
            elif a == "--level" and i + 1 < len(args):
                level = args[i + 1].upper()
                
        # Get log file path
        log_file = os.path.join(LOG_DIR, "legion_agi.log")
        if not os.path.exists(log_file):
            console.print(f"Log file not found: {log_file}", style="yellow")
            return
            
        # Read log lines
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
            # Filter by level if specified
            if level:
                lines = [line for line in lines if f"| {level} |" in line]
                
            # Get last N lines
            lines = lines[-num_lines:]
            
            # Display logs with colored levels
            console.print(f"Last {len(lines)} log entries:", style="blue")
            
            for line in lines:
                # Extract level for coloring
                if "| DEBUG |" in line:
                    styled_line = Text(line.rstrip())
                    styled_line.stylize("dim")
                elif "| INFO |" in line:
                    styled_line = Text(line.rstrip())
                    styled_line.stylize("blue")
                elif "| WARNING |" in line:
                    styled_line = Text(line.rstrip())
                    styled_line.stylize("yellow")
                elif "| ERROR |" in line:
                    styled_line = Text(line.rstrip())
                    styled_line.stylize("red")
                elif "| CRITICAL |" in line:
                    styled_line = Text(line.rstrip())
                    styled_line.stylize("red bold")
                else:
                    styled_line = Text(line.rstrip())
                    
                console.print(styled_line)
                
        except Exception as e:
            console.print(f"Error reading logs: {e}", style="red")
            
    def do_help(self, arg: str) -> None:
        """Display help for commands."""
        if arg:
            # Help for specific command
            super().do_help(arg)
            return
            
        # Custom help display
        console.print("\nLegion AGI Command Reference:", style="bold blue")
        
        # Create help table
        table = Table(title="Available Commands")
        table.add_column("Command", style="cyan")
        table.add_column("Description", style="green")
        table.add_column("Usage Example", style="yellow")
        
        # System commands
        table.add_row("start", "Initialize and start the Legion AGI system", 
                    "start [--load path/to/state]")
        table.add_row("status", "Display current system status", "status")
        table.add_row("exit / quit", "Exit the CLI and shutdown the system", "exit")
        
        # Query commands
        table.add_row("query", "Process a query with Legion AGI", 
                    "query How does quantum computing work?")
        table.add_row("/past", "Use PAST method for a query", 
                    "/past Explain consciousness theories")
        table.add_row("/raft", "Use RAFT method for a query", 
                    "/raft Design a sustainable energy system")
        table.add_row("/eat", "Use EAT method for a query", 
                    "/eat Evaluate approaches to AGI safety")
        table.add_row("method", "Set the default reasoning method", "method raft")
        
        # System information
        table.add_row("agents", "Display information about current agents", "agents")
        table.add_row("viz", "Open visualization files", "viz [agents|memory|workspace|all]")
        table.add_row("logs", "Display recent log entries", "logs 50 --level ERROR")
        table.add_row("save", "Save current system state", "save [filename]")
        
        console.print(table)
        
        console.print("\nNote: You can also directly type your question without the 'query' command.", style="blue")
        console.print("Press Tab for command auto-completion.", style="blue")
        
    def _display_status(self, status: Dict[str, Any]) -> None:
        """
        Display system status in a formatted panel.
        
        Args:
            status: System status dictionary
        """
        # Create status table
        table = Table(title="Legion AGI System Status")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        # Add rows for each status item
        for key, value in status.items():
            if key == "agent_types" and isinstance(value, list):
                table.add_row(key, ", ".join(value))
            else:
                table.add_row(key, str(value))
                
        console.print(table)
        
    def _display_result(self, result: Dict[str, Any]) -> None:
        """
        Display query result in a formatted panel.
        
        Args:
            result: Query result dictionary
        """
        if "error" in result:
            console.print(Panel(f"Error: {result['error']}", border_style="red"))
            return
            
        if "integrated_solution" in result:
            # Check if solution looks like markdown
            solution = result["integrated_solution"]
            if "```" in solution or "#" in solution or "*" in solution:
                # Format as markdown
                console.print(Markdown(solution))
            else:
                # Regular panel
                console.print(Panel(solution, border_style="green"))
                
            # Show contributors if available
            if "contributors" in result and isinstance(result["contributors"], list):
                contrib_names = [f"{c.get('name', 'Unknown')} ({c.get('role', 'Unknown')})" 
                                for c in result["contributors"]]
                console.print("Contributors:", style="blue")
                for name in contrib_names:
                    console.print(f"- {name}")
        else:
            # Just print raw result
            console.print(result)
            
    def _run_evolution_thread(self) -> None:
        """Background thread for running evolution mode."""
        try:
            if self.legion:
                self.legion._run_evolution_mode()
        except Exception as e:
            logger.exception(f"Error in evolution thread: {e}")
            console.print(f"\nError in evolution process: {e}", style="red")


def main():
    """Main entry point for Legion CLI."""
    parser = argparse.ArgumentParser(description="Legion AGI Command Line Interface")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                       help="LLM model to use")
    parser.add_argument("--mode", type=str, default=TOOL_MODE,
                       choices=[TOOL_MODE, EVOLUTION_MODE],
                       help="Operation mode (tool or evolve)")
    parser.add_argument("--session", type=str, default=None,
                       help="Session ID for persistence")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                       help="Directory for data storage")
    parser.add_argument("--no-viz", action="store_true",
                       help="Disable visualizations")
    parser.add_argument("--query", type=str, default=None,
                       help="Process a single query and exit")
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = LegionCLI(
        model=args.model,
        mode=args.mode,
        session_id=args.session,
        data_dir=args.data_dir,
        visualize=not args.no_viz
    )
    
    # Process single query if provided
    if args.query:
        cli.do_start("")
        if cli.legion and cli.legion.running:
            cli.do_query(args.query)
            cli.do_exit("")
        return
        
    # Start interactive mode
    try:
        cli.cmdloop()
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt. Exiting...")
    except Exception as e:
        logger.exception(f"Error in CLI: {e}")
        print(f"Error in CLI: {e}")
    finally:
        # Make sure to clean up
        if cli.legion and cli.legion.running:
            cli.legion.stop()


if __name__ == "__main__":
    main()

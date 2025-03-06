"""
Visualization Tools for Legion AGI System

This module provides visualization capabilities for the Legion AGI system,
helping to visualize thought processes, memory states, and agent interactions.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from legion_agi.core.memory_module import MemoryModule
from legion_agi.agents.agent_base import Agent
from legion_agi.core.global_workspace import GlobalWorkspace


class SystemVisualizer:
    """
    Visualization tools for the Legion AGI system.
    Provides methods to visualize system components and interactions.
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize the system visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Color maps for different types of visualizations
        self.agent_colors = plt.cm.tab10(np.linspace(0, 1, 10))
        self.memory_colors = {
            "stm": "#3498db",   # Blue
            "ltm": "#2ecc71",   # Green
            "episodic": "#e74c3c", # Red
            "semantic": "#9b59b6", # Purple
            "working": "#f39c12"   # Orange
        }
        
        # Interaction counter for filename generation
        self.interaction_counter = 0
        
        logger.info(f"System visualizer initialized with output directory: {output_dir}")
        
    def visualize_agent_network(self, 
                               agents: List[Agent], 
                               filename: Optional[str] = None) -> str:
        """
        Visualize the network of agents and their interactions.
        
        Args:
            agents: List of agents to visualize
            filename: Optional filename for the visualization
            
        Returns:
            Path to the saved visualization file
        """
        # Create graph
        G = nx.Graph()
        
        # Add agent nodes
        for i, agent in enumerate(agents):
            G.add_node(agent.name, 
                      role=agent.role, 
                      color=self.agent_colors[i % len(self.agent_colors)],
                      size=300)  # Base size
            
        # Add connections between agents
        for i, agent in enumerate(agents):
            for other_agent in agent.agent_list:
                if other_agent.name != agent.name:
                    G.add_edge(agent.name, other_agent.name)
                    
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Create position layout
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Get node colors and sizes
        node_colors = [G.nodes[node]['color'] for node in G.nodes()]
        node_sizes = [G.nodes[node]['size'] for node in G.nodes()]
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.5, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        # Add role labels
        role_labels = {node: G.nodes[node]['role'] for node in G.nodes()}
        pos_roles = {k: (v[0], v[1] - 0.08) for k, v in pos.items()}
        nx.draw_networkx_labels(G, pos_roles, labels=role_labels, font_size=10, font_color='darkblue')
        
        # Set title and layout
        plt.title("Legion AGI Agent Network", fontsize=16, fontweight='bold')
        plt.axis('off')
        
        # Set default filename if not provided
        if filename is None:
            filename = f"agent_network_{len(agents)}_agents.png"
            
        # Save figure
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Agent network visualization saved to {filepath}")
        return filepath
        
    def visualize_memory_state(self, 
                              memory_module: MemoryModule, 
                              filename: Optional[str] = None) -> str:
        """
        Visualize the state of a memory module.
        
        Args:
            memory_module: Memory module to visualize
            filename: Optional filename for the visualization
            
        Returns:
            Path to the saved visualization file
        """
        # Memory counts for each type
        memory_counts = {
            "Short-term Memory": len(memory_module.short_term_memory),
            "Long-term Memory": len(memory_module.long_term_memory),
            "Episodic Memory": len(memory_module.episodic_memory),
            "Semantic Memory": len(memory_module.semantic_memory),
            "Working Memory": len(memory_module.working_memory)
        }
        
        # Memory importance distributions
        stm_importance = [m['importance'] for m in memory_module.short_term_memory if 'importance' in m]
        ltm_importance = [m['importance'] for m in memory_module.long_term_memory if 'importance' in m]
        
        # Create visualization with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Plot memory counts
        bars = ax1.bar(memory_counts.keys(), memory_counts.values(), 
                     color=[self.memory_colors['stm'], self.memory_colors['ltm'], 
                            self.memory_colors['episodic'], self.memory_colors['semantic'], 
                            self.memory_colors['working']])
        
        # Add count labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
                    
        ax1.set_title("Memory Size by Type", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Number of Memory Items")
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot importance distributions if data available
        if stm_importance or ltm_importance:
            if stm_importance:
                ax2.hist(stm_importance, bins=10, alpha=0.7, label="STM Importance", 
                        color=self.memory_colors['stm'])
            if ltm_importance:
                ax2.hist(ltm_importance, bins=10, alpha=0.7, label="LTM Importance", 
                        color=self.memory_colors['ltm'])
                        
            ax2.set_title("Memory Importance Distribution", fontsize=14, fontweight='bold')
            ax2.set_xlabel("Importance Value")
            ax2.set_ylabel("Frequency")
            ax2.legend()
            ax2.grid(linestyle='--', alpha=0.7)
        else:
            # If no importance data, show memory types distribution
            memory_types = {
                "Response": 0,
                "Reasoning": 0,
                "Collaboration": 0,
                "External": 0,
                "Other": 0
            }
            
            # Count memory types
            for memory in memory_module.short_term_memory + memory_module.long_term_memory:
                if 'metadata' in memory and 'type' in memory['metadata']:
                    mem_type = memory['metadata']['type']
                    if mem_type == 'response':
                        memory_types["Response"] += 1
                    elif mem_type == 'reasoning_step':
                        memory_types["Reasoning"] += 1
                    elif mem_type == 'collaboration':
                        memory_types["Collaboration"] += 1
                    elif mem_type == 'external':
                        memory_types["External"] += 1
                    else:
                        memory_types["Other"] += 1
                else:
                    memory_types["Other"] += 1
                    
            # Plot memory types
            ax2.pie(memory_types.values(), labels=memory_types.keys(), autopct='%1.1f%%',
                   shadow=True, startangle=90, colors=plt.cm.Paired(np.linspace(0, 1, len(memory_types))))
            ax2.axis('equal')
            ax2.set_title("Memory Content Types", fontsize=14, fontweight='bold')
            
        # Set default filename if not provided
        if filename is None:
            filename = f"memory_state_{memory_module.name}.png"
            
        # Set main title
        plt.suptitle(f"Memory State: {memory_module.name}", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Memory state visualization saved to {filepath}")
        return filepath
        
    def visualize_agent_interaction(self, 
                                   conversation_history: List[Dict[str, str]], 
                                   filename: Optional[str] = None) -> str:
        """
        Visualize agent interactions from conversation history.
        
        Args:
            conversation_history: Conversation history to visualize
            filename: Optional filename for the visualization
            
        Returns:
            Path to the saved visualization file
        """
        # Extract agents and messages
        agents = set()
        interactions = []
        
        for entry in conversation_history:
            role = entry.get('role', 'Unknown')
            content = entry.get('content', '')
            agents.add(role)
            
            # Only count substantial messages
            if len(content) > 50:
                interactions.append((role, len(content)))
                
        # Create directed graph
        G = nx.DiGraph()
        
        # Add agent nodes
        agent_list = list(agents)
        for i, agent in enumerate(agent_list):
            G.add_node(agent, color=self.agent_colors[i % len(self.agent_colors)], 
                      size=200 + 100 * interactions.count((agent, interactions)))
                      
        # Add interaction edges
        for i in range(len(interactions) - 1):
            source = interactions[i][0]
            target = interactions[i + 1][0]
            
            if source != target:
                # Check if edge already exists
                if G.has_edge(source, target):
                    # Increment weight
                    G[source][target]['weight'] += 1
                else:
                    # Create new edge
                    G.add_edge(source, target, weight=1)
                    
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Create position layout
        pos = nx.spring_layout(G, k=0.3, iterations=50)
        
        # Get node attributes
        node_colors = [G.nodes[node]['color'] for node in G.nodes()]
        
        # Calculate node sizes based on message count
        node_sizes = []
        for node in G.nodes():
            count = sum(1 for interaction in interactions if interaction[0] == node)
            size = 500 + 100 * count
            node_sizes.append(size)
            
        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        
        # Draw edges with varying width based on weight
        edge_widths = [G[u][v]['weight'] * 2 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7, 
                              edge_color='gray', arrows=True, arrowsize=20)
                              
        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        # Add edge labels (weights)
        edge_labels = {(u, v): G[u][v]['weight'] for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
        
        # Set title and layout
        plt.title("Agent Interaction Pattern", fontsize=16, fontweight='bold')
        plt.axis('off')
        
        # Set default filename if not provided
        if filename is None:
            self.interaction_counter += 1
            filename = f"agent_interaction_{self.interaction_counter}.png"
            
        # Save figure
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Agent interaction visualization saved to {filepath}")
        return filepath
        
    def visualize_global_workspace(self, 
                                  workspace: GlobalWorkspace, 
                                  filename: Optional[str] = None) -> str:
        """
        Visualize the state of the global workspace.
        
        Args:
            workspace: Global workspace to visualize
            filename: Optional filename for the visualization
            
        Returns:
            Path to the saved visualization file
        """
        # Extract workspace data
        current_contents = workspace.get_workspace_contents()
        broadcast_history = workspace.get_broadcast_history()
        specialists = workspace.specialists
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Define grid layout
        gs = plt.GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])  # Current contents
        ax2 = fig.add_subplot(gs[0, 1])  # Broadcast history
        ax3 = fig.add_subplot(gs[1, :])  # Specialist activity
        
        # Plot current workspace contents
        if current_contents:
            # Group by source
            sources = {}
            for info in current_contents:
                source = info.source
                if source in sources:
                    sources[source] += 1
                else:
                    sources[source] = 1
                    
            # Plot as pie chart
            ax1.pie(sources.values(), labels=sources.keys(), autopct='%1.1f%%',
                   shadow=True, startangle=90, colors=plt.cm.tab20(np.linspace(0, 1, len(sources))))
            ax1.axis('equal')
            ax1.set_title("Current Workspace Contents by Source", fontsize=14)
        else:
            ax1.text(0.5, 0.5, "No current contents in workspace", 
                    ha='center', va='center', fontsize=12)
            ax1.axis('off')
            
        # Plot broadcast history
        if broadcast_history:
            # Count broadcasts per cycle
            broadcast_counts = [len(broadcast) for broadcast in broadcast_history]
            ax2.plot(range(1, len(broadcast_counts) + 1), broadcast_counts, 
                    marker='o', linestyle='-', linewidth=2, markersize=8)
            ax2.set_xlabel("Workspace Cycle")
            ax2.set_ylabel("Number of Information Units")
            ax2.set_title("Broadcast History", fontsize=14)
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Add trend line
            if len(broadcast_counts) > 2:
                z = np.polyfit(range(1, len(broadcast_counts) + 1), broadcast_counts, 1)
                p = np.poly1d(z)
                ax2.plot(range(1, len(broadcast_counts) + 1), p(range(1, len(broadcast_counts) + 1)), 
                        "r--", alpha=0.8)
        else:
            ax2.text(0.5, 0.5, "No broadcast history available", 
                    ha='center', va='center', fontsize=12)
            ax2.axis('off')
            
        # Plot specialist activity
        if specialists:
            # Create data for specialist activity
            specialist_names = list(specialists.keys())
            
            # For demonstration, create random activity levels
            # In a real implementation, we would extract actual activity metrics
            import random
            activity_levels = [specialists[name].activation for name in specialist_names]
            
            # Sort by activity level
            sorted_indices = np.argsort(activity_levels)
            specialist_names = [specialist_names[i] for i in sorted_indices]
            activity_levels = [activity_levels[i] for i in sorted_indices]
            
            # Plot horizontal bar chart
            bars = ax3.barh(specialist_names, activity_levels, 
                           color=plt.cm.viridis(np.linspace(0, 1, len(specialist_names))))
                           
            # Add value labels
            for i, v in enumerate(activity_levels):
                ax3.text(v + 0.01, i, f"{v:.2f}", va='center')
                
            ax3.set_xlim(0, 1.1)
            ax3.set_xlabel("Activation Level")
            ax3.set_title("Specialist Module Activation Levels", fontsize=14)
            ax3.grid(True, axis='x', linestyle='--', alpha=0.7)
        else:
            ax3.text(0.5, 0.5, "No specialist modules available", 
                    ha='center', va='center', fontsize=12)
            ax3.axis('off')
            
        # Set default filename if not provided
        if filename is None:
            filename = f"global_workspace_state.png"
            
        # Set main title
        plt.suptitle("Global Workspace State", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Global workspace visualization saved to {filepath}")
        return filepath
        
    def visualize_reasoning_process(self, 
                                   reasoning_chain: List[Dict[str, Any]], 
                                   filename: Optional[str] = None) -> str:
        """
        Visualize a reasoning process from a reasoning chain.
        
        Args:
            reasoning_chain: List of reasoning steps
            filename: Optional filename for the visualization
            
        Returns:
            Path to the saved visualization file
        """
        if not reasoning_chain:
            logger.warning("Empty reasoning chain provided for visualization")
            return ""
            
        # Extract reasoning data
        steps = len(reasoning_chain)
        iterations = [step.get('iteration', i+1) for i, step in enumerate(reasoning_chain)]
        
        # Extract step lengths and create complexity metric
        step_lengths = [len(step.get('reasoning_output', '')) for step in reasoning_chain]
        complexity = [min(len(set(step.get('reasoning_output', '').split())) / 100, 1.0) 
                     for step in reasoning_chain]
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot step length progression
        ax1.plot(iterations, step_lengths, marker='o', linestyle='-', 
                linewidth=2, markersize=10, label='Step Length')
        ax1.set_xlabel("Reasoning Iteration", fontsize=12)
        ax1.set_ylabel("Step Length (characters)", fontsize=12)
        ax1.set_title("Reasoning Process Progression", fontsize=14, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add annotations for key points
        max_idx = step_lengths.index(max(step_lengths))
        ax1.annotate(f"Peak: {step_lengths[max_idx]} chars",
                    xy=(iterations[max_idx], step_lengths[max_idx]),
                    xytext=(iterations[max_idx], step_lengths[max_idx] + 100),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=10, ha='center')
                    
        # Plot complexity metric
        ax1.plot(iterations, [c * max(step_lengths) for c in complexity], 
                marker='s', linestyle='--', linewidth=2, markersize=8,
                color='red', label='Complexity')
        ax1.legend(fontsize=10)
        
        # Create reasoning flow diagram in bottom subplot
        # Use a directed graph to show the flow
        G = nx.DiGraph()
        
        # Add nodes for each reasoning step
        for i, step in enumerate(reasoning_chain):
            G.add_node(i, step=iterations[i], 
                      output=step.get('reasoning_output', '')[:50] + '...')
                      
        # Add edges between steps
        for i in range(len(reasoning_chain) - 1):
            G.add_edge(i, i+1)
            
        # Draw the graph
        pos = {i: (i, 0) for i in range(len(reasoning_chain))}
        nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=1000, 
                              node_color=plt.cm.YlOrRd(np.linspace(0, 1, len(reasoning_chain))))
        nx.draw_networkx_edges(G, pos, ax=ax2, arrows=True, 
                              arrowsize=20, width=2, edge_color='gray')
                              
        # Add step labels
        labels = {i: f"Step {G.nodes[i]['step']}" for i in G.nodes()}
        nx.draw_networkx_labels(G, pos, ax=ax2, labels=labels, font_size=10, font_weight='bold')
        
        # Add mini step descriptions
        for i in G.nodes():
            ax2.text(i, -0.15, G.nodes[i]['output'], 
                    ha='center', va='top', fontsize=8, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                    
        ax2.set_title("Reasoning Flow", fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Set main title
        goal = reasoning_chain[0].get('goal', 'Unknown Goal')
        plt.suptitle(f"Reasoning Process: {goal[:50]}{'...' if len(goal) > 50 else ''}", 
                    fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Set default filename if not provided
        if filename is None:
            filename = f"reasoning_process_{steps}_steps.png"
            
        # Save figure
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Reasoning process visualization saved to {filepath}")
        return filepath

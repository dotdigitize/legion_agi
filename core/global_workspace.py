"""
Global Workspace Architecture

Implements the Global Workspace Theory (GWT) architecture, which suggests that
consciousness emerges from a central information exchange where specialized
neural processors compete for access to a shared global workspace.

This module simulates this architecture for the Legion AGI system.
"""

import numpy as np
import heapq
from typing import Dict, List, Tuple, Any, Optional, Callable
from loguru import logger

from legion_agi.config import (
    GW_COMPETITION_THRESHOLD, 
    GW_BROADCAST_CYCLES,
    GW_WORKSPACE_CAPACITY
)


class InformationUnit:
    """
    Information unit for the Global Workspace.
    Represents a discrete unit of information that can be broadcast.
    """
    
    def __init__(self, 
                content: Any, 
                source: str, 
                activation: float = 0.0,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize information unit.
        
        Args:
            content: The actual information content (can be any type)
            source: Source module or agent that generated this information
            activation: Initial activation level (0.0 to 1.0)
            metadata: Additional metadata about this information
        """
        self.id = id(self)  # Unique identifier
        self.content = content
        self.source = source
        self.activation = activation
        self.metadata = metadata or {}
        self.creation_time = np.datetime64('now')
        self.broadcast_count = 0
        
    def __lt__(self, other):
        """
        Compare information units by activation level for priority queue.
        """
        return self.activation > other.activation  # Higher activation = higher priority


class SpecialistModule:
    """
    Specialist processor module in Global Workspace architecture.
    Represents a specialized cognitive module that can process and generate information.
    """
    
    def __init__(self, 
                name: str, 
                process_function: Callable[[List[InformationUnit]], List[InformationUnit]]):
        """
        Initialize specialist module.
        
        Args:
            name: Name of the specialist module
            process_function: Function that processes input information units and returns new ones
        """
        self.name = name
        self.process_function = process_function
        self.input_buffer: List[InformationUnit] = []
        self.output_buffer: List[InformationUnit] = []
        self.activation = 0.5  # Initial activation level
        
    def receive_broadcast(self, information_units: List[InformationUnit]) -> None:
        """
        Receive broadcast information from global workspace.
        
        Args:
            information_units: List of information units from broadcast
        """
        self.input_buffer.extend(information_units)
        
    def process(self) -> List[InformationUnit]:
        """
        Process information in input buffer and generate output.
        
        Returns:
            List of new information units generated by this module
        """
        if not self.input_buffer:
            return []
            
        # Process information using provided function
        new_information = self.process_function(self.input_buffer)
        
        # Tag information with this module as source
        for info in new_information:
            info.source = self.name
            
        # Update output buffer
        self.output_buffer.extend(new_information)
        
        # Clear input buffer
        self.input_buffer = []
        
        return new_information
    
    def compete(self) -> Tuple[List[InformationUnit], float]:
        """
        Compete for access to global workspace.
        
        Returns:
            Tuple of (information units to send to workspace, combined activation)
        """
        if not self.output_buffer:
            return [], 0.0
            
        # Calculate combined activation for competition
        # This could be based on various factors (importance, relevance, etc.)
        combined_activation = self.activation * max(info.activation for info in self.output_buffer)
        
        # Return information and combined activation
        result = self.output_buffer
        self.output_buffer = []
        return result, combined_activation


class GlobalWorkspace:
    """
    Central Global Workspace implementation.
    Simulates the global workspace where information is broadcast to all specialist modules.
    """
    
    def __init__(self):
        """Initialize Global Workspace."""
        self.specialists: Dict[str, SpecialistModule] = {}
        self.contents: List[InformationUnit] = []
        self.broadcast_history: List[List[InformationUnit]] = []
        self.competition_threshold = GW_COMPETITION_THRESHOLD
        self.broadcast_cycles = GW_BROADCAST_CYCLES
        self.capacity = GW_WORKSPACE_CAPACITY
        
    def register_specialist(self, specialist: SpecialistModule) -> None:
        """
        Register a specialist module with the global workspace.
        
        Args:
            specialist: Specialist module to register
        """
        self.specialists[specialist.name] = specialist
        logger.info(f"Registered specialist module: {specialist.name}")
        
    def broadcast(self) -> None:
        """
        Broadcast current workspace contents to all specialist modules.
        """
        if not self.contents:
            return
            
        # Log broadcast
        logger.info(f"Broadcasting {len(self.contents)} information units to {len(self.specialists)} specialists")
        
        # Update broadcast count
        for info in self.contents:
            info.broadcast_count += 1
            
        # Store in history
        self.broadcast_history.append(self.contents[:])
        
        # Limit history size
        if len(self.broadcast_history) > 100:
            self.broadcast_history.pop(0)
            
        # Broadcast to all specialists
        for specialist in self.specialists.values():
            specialist.receive_broadcast(self.contents)
            
    def run_competition(self) -> None:
        """
        Run competition for access to the global workspace.
        Specialists compete to have their information broadcast.
        """
        # Gather all competing information
        competition_entries: List[Tuple[List[InformationUnit], float, str]] = []
        
        for name, specialist in self.specialists.items():
            info_units, activation = specialist.compete()
            if info_units and activation >= self.competition_threshold:
                competition_entries.append((info_units, activation, name))
                
        if not competition_entries:
            logger.info("No specialists competed for workspace access")
            return
            
        # Sort by activation level (highest first)
        competition_entries.sort(key=lambda x: x[1], reverse=True)
        
        # Select winners up to capacity
        new_contents = []
        winners = []
        
        for info_units, activation, name in competition_entries:
            if len(new_contents) + len(info_units) <= self.capacity:
                new_contents.extend(info_units)
                winners.append(name)
            else:
                # Take as many as possible up to capacity
                remaining = self.capacity - len(new_contents)
                if remaining > 0:
                    # Sort by activation and take highest
                    sorted_units = sorted(info_units, key=lambda x: x.activation, reverse=True)
                    new_contents.extend(sorted_units[:remaining])
                    winners.append(name)
                break
                
        if winners:
            logger.info(f"Competition winners: {', '.join(winners)}")
            self.contents = new_contents
        else:
            logger.info("No information selected in competition")
            
    def process_cycle(self) -> None:
        """
        Run a complete workspace cycle:
        1. Broadcast current contents
        2. Let specialists process information
        3. Run competition for next broadcast
        """
        # Broadcast current contents
        self.broadcast()
        
        # Let specialists process information
        for specialist in self.specialists.values():
            specialist.process()
            
        # Run competition for next workspace access
        self.run_competition()
        
    def run_cycles(self, num_cycles: int = GW_BROADCAST_CYCLES) -> None:
        """
        Run multiple workspace cycles.
        
        Args:
            num_cycles: Number of cycles to run
        """
        for i in range(num_cycles):
            logger.info(f"Running Global Workspace cycle {i+1}/{num_cycles}")
            self.process_cycle()
            
    def inject_information(self, 
                          content: Any, 
                          source: str, 
                          activation: float = 0.8,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Inject external information directly into the workspace.
        
        Args:
            content: Information content
            source: Source of the information
            activation: Activation level (0.0 to 1.0)
            metadata: Additional metadata
        """
        info_unit = InformationUnit(content, source, activation, metadata)
        
        # If we have space, add directly
        if len(self.contents) < self.capacity:
            self.contents.append(info_unit)
        else:
            # Replace lowest activation item if this one is higher
            lowest_idx = min(range(len(self.contents)), 
                           key=lambda i: self.contents[i].activation)
            
            if self.contents[lowest_idx].activation < activation:
                self.contents[lowest_idx] = info_unit
                
        logger.info(f"Injected information from {source} into workspace")
        
    def get_workspace_contents(self) -> List[InformationUnit]:
        """
        Get current contents of the workspace.
        
        Returns:
            List of information units currently in the workspace
        """
        return self.contents[:]
        
    def clear_workspace(self) -> None:
        """Clear all contents from the workspace."""
        self.contents = []
        logger.info("Cleared Global Workspace contents")
        
    def get_broadcast_history(self) -> List[List[InformationUnit]]:
        """
        Get history of all broadcasts.
        
        Returns:
            List of broadcast information sets
        """
        return self.broadcast_history[:]


class AttentionalBottleneck:
    """
    Attentional bottleneck component for the Global Workspace.
    Implements attention mechanisms to filter and prioritize information.
    """
    
    def __init__(self, global_workspace: GlobalWorkspace, capacity: int = 3):
        """
        Initialize attentional bottleneck.
        
        Args:
            global_workspace: Global workspace to connect to
            capacity: Maximum number of items that can pass through bottleneck
        """
        self.global_workspace = global_workspace
        self.capacity = capacity
        self.priority_queue = []  # Priority queue for information units
        self.attention_filters: List[Callable[[InformationUnit], float]] = []
        
    def add_attention_filter(self, 
                            filter_function: Callable[[InformationUnit], float]) -> None:
        """
        Add attention filter function.
        
        Args:
            filter_function: Function that takes information unit and returns attention score (0-1)
        """
        self.attention_filters.append(filter_function)
        
    def filter_information(self, information_units: List[InformationUnit]) -> List[InformationUnit]:
        """
        Filter information based on attention filters.
        
        Args:
            information_units: List of information units to filter
            
        Returns:
            Filtered list of information units
        """
        if not self.attention_filters:
            return information_units[:self.capacity]  # Just limit by capacity
            
        # Calculate attention scores for each information unit
        scored_units = []
        
        for unit in information_units:
            # Apply all filters and take average score
            scores = [filter_fn(unit) for filter_fn in self.attention_filters]
            avg_score = sum(scores) / len(scores) if scores else 0.5
            unit.activation = avg_score  # Update activation based on attention
            scored_units.append((unit, avg_score))
            
        # Sort by score (highest first)
        scored_units.sort(key=lambda x: x[1], reverse=True)
        
        # Return top items up to capacity
        return [unit for unit, _ in scored_units[:self.capacity]]
        
    def attend(self, information_units: List[InformationUnit]) -> None:
        """
        Apply attention to information units and pass to global workspace.
        
        Args:
            information_units: List of information units to attend to
        """
        # Apply attention filters
        filtered_units = self.filter_information(information_units)
        
        # Add to global workspace
        for unit in filtered_units:
            self.global_workspace.inject_information(
                unit.content, unit.source, unit.activation, unit.metadata
            )

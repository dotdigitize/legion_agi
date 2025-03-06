"""
Memory Module integrating quantum, neural, and symbolic memory systems.

This module provides a unified memory architecture that integrates multiple
memory types, including quantum memory, spiking neural networks, and 
traditional symbolic storage. It supports memory consolidation, retrieval,
and the simulation of cognitive memory processes.
"""

import numpy as np
import json
import pickle
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from loguru import logger

from legion_agi.core.quantum_memory import QuantumMemory
from legion_agi.core.spiking_neurons import SpikingNeuralNetwork
from legion_agi.config import (
    AGENT_MEMORY_CAPACITY,
    CONSOLIDATION_THRESHOLD
)


class MemoryModule:
    """
    Integrated memory system combining quantum, neural, and symbolic memory.
    """
    
    def __init__(self, 
                name: str,
                stm_capacity: int = AGENT_MEMORY_CAPACITY,
                enable_quantum: bool = True,
                enable_snn: bool = True):
        """
        Initialize memory module.
        
        Args:
            name: Name of this memory module
            stm_capacity: Short-term memory capacity
            enable_quantum: Whether to enable quantum memory simulation
            enable_snn: Whether to enable spiking neural network simulation
        """
        self.name = name
        self.id = str(uuid.uuid4())
        
        # Memory systems
        self.short_term_memory: List[Dict[str, Any]] = []
        self.long_term_memory: List[Dict[str, Any]] = []
        self.episodic_memory: List[Dict[str, Any]] = []
        self.semantic_memory: Dict[str, Any] = {}
        self.procedural_memory: Dict[str, Any] = {}
        self.working_memory: Dict[str, Any] = {}
        
        # Replay buffer for memory consolidation
        self.replay_buffer: List[Dict[str, Any]] = []
        
        # Memory capacities
        self.stm_capacity = stm_capacity
        self.ltm_capacity = stm_capacity * 10
        self.episodic_capacity = stm_capacity * 5
        
        # Memory systems
        self.quantum_memory = QuantumMemory() if enable_quantum else None
        self.snn_memory = SpikingNeuralNetwork() if enable_snn else None
        
        # Memory maps to track quantum/neural representations
        self.quantum_memory_map: Dict[str, str] = {}
        self.neural_memory_map: Dict[str, Dict[str, Any]] = {}
        
        # Init specialized memory structures if neural networks enabled
        if self.snn_memory is not None:
            self._init_neural_memory_structures()
            
        # Associative networks
        self.associations: Dict[str, Set[str]] = {}
        
        logger.info(f"Memory module '{name}' initialized with "
                   f"quantum={enable_quantum}, snn={enable_snn}")
        
    def _init_neural_memory_structures(self) -> None:
        """Initialize specialized neural memory structures."""
        # Create working memory circuit
        self.snn_memory.create_working_memory_circuit(
            f"{self.name}_working",
            num_items=7,  # Miller's Law
            neurons_per_item=50
        )
        
        # Create episodic memory circuit
        self.snn_memory.create_hippocampal_memory(
            f"{self.name}_episodic",
            num_patterns=100,
            pattern_size=50
        )
        
        # Create semantic memory network
        self.snn_memory.create_neuronal_memory(
            f"{self.name}_semantic",
            memory_size=500,
            inhibition_strength=0.8
        )
        
    def _generate_memory_id(self, content: Any) -> str:
        """
        Generate a unique ID for a memory item.
        
        Args:
            content: Memory content to generate ID for
            
        Returns:
            Unique memory ID
        """
        timestamp = datetime.now().isoformat()
        content_hash = hash(str(content))
        return f"{self.name}_{timestamp}_{content_hash}"
        
    def add_to_stm(self, 
                 content: Any, 
                 source: str, 
                 metadata: Optional[Dict[str, Any]] = None,
                 importance: float = 0.5) -> str:
        """
        Add memory item to short-term memory.
        
        Args:
            content: Memory content (can be any serializable type)
            source: Source of the memory (e.g., agent name, perception)
            metadata: Additional metadata about the memory
            importance: Importance of the memory (0.0 to 1.0)
            
        Returns:
            Memory ID
        """
        # Generate memory ID
        memory_id = self._generate_memory_id(content)
        
        # Create memory item
        memory_item = {
            'id': memory_id,
            'content': content,
            'source': source,
            'timestamp': datetime.now().isoformat(),
            'importance': importance,
            'metadata': metadata or {},
            'access_count': 0,
            'last_accessed': datetime.now().isoformat()
        }
        
        # Add to short-term memory
        self.short_term_memory.append(memory_item)
        
        # Also add a reference copy to the replay buffer
        self.replay_buffer.append(memory_item.copy())
        
        # Limit short-term memory size
        if len(self.short_term_memory) > self.stm_capacity:
            # Remove least important item
            self.short_term_memory.sort(key=lambda x: x['importance'])
            removed = self.short_term_memory.pop(0)
            logger.debug(f"STM capacity reached, removed memory: {removed['id']}")
            
        # Add to quantum memory if enabled
        if self.quantum_memory is not None:
            self._store_in_quantum_memory(memory_item)
            
        # Add to neural memory if enabled
        if self.snn_memory is not None:
            self._encode_in_neural_memory(memory_item, "stm")
            
        # Check if consolidation needed
        if len(self.short_term_memory) >= CONSOLIDATION_THRESHOLD:
            self.consolidate_memory()
            
        logger.debug(f"Added memory {memory_id} to short-term memory")
        return memory_id
        
    def _store_in_quantum_memory(self, memory_item: Dict[str, Any]) -> None:
        """
        Store memory in quantum memory system.
        
        Args:
            memory_item: Memory item to store
        """
        # For text content, store directly
        if isinstance(memory_item['content'], str):
            quantum_id = self.quantum_memory.store_memory(
                memory_item['content'], 
                "working"
            )
            
        # For structured content, serialize to JSON
        elif isinstance(memory_item['content'], (dict, list)):
            serialized = json.dumps(memory_item['content'])
            quantum_id = self.quantum_memory.store_memory(
                serialized,
                "working"
            )
            
        # For other types, try to convert to string
        else:
            try:
                serialized = str(memory_item['content'])
                quantum_id = self.quantum_memory.store_memory(
                    serialized,
                    "working"
                )
            except Exception as e:
                logger.warning(f"Could not store in quantum memory: {e}")
                return
                
        # Store mapping from memory ID to quantum ID
        self.quantum_memory_map[memory_item['id']] = quantum_id
        
    def _encode_in_neural_memory(self, 
                               memory_item: Dict[str, Any],
                               memory_type: str) -> None:
        """
        Encode memory in spiking neural network.
        
        Args:
            memory_item: Memory item to encode
            memory_type: Type of memory ("stm", "ltm", "episodic", "semantic")
        """
        # Determine which neural circuit to use
        if memory_type == "stm" or memory_type == "working":
            circuit_name = f"{self.name}_working"
        elif memory_type == "episodic":
            circuit_name = f"{self.name}_episodic"
        elif memory_type == "semantic" or memory_type == "ltm":
            circuit_name = f"{self.name}_semantic"
        else:
            logger.warning(f"Unknown neural memory type: {memory_type}")
            return
            
        # Convert memory content to a numerical representation
        encoding = self._neural_encode_content(memory_item['content'])
        
        # Encode in appropriate neural circuit
        if circuit_name.endswith("_working"):
            # Find available item slot and encode
            for i in range(7):  # 7 slots in working memory
                start_idx = i * 50  # 50 neurons per item
                indices = list(range(start_idx, start_idx + 50))
                
                # Check if slot is empty (no current)
                ng = self.snn_memory.neuron_groups[circuit_name]
                if np.sum(ng.I[indices]) < 0.1:
                    # Encode in this slot
                    self.snn_memory.set_input_current(
                        circuit_name, 
                        indices[:len(encoding)], 
                        encoding * memory_item['importance']
                    )
                    
                    # Store mapping
                    self.neural_memory_map[memory_item['id']] = {
                        'circuit': circuit_name,
                        'indices': indices,
                        'type': memory_type
                    }
                    break
                    
        elif circuit_name.endswith("_episodic"):
            # Find DG layer and encode
            dg_name = f"{self.name}_episodic_DG"
            pattern = self._hash_to_binary_pattern(
                str(memory_item['content']), 
                len(self.snn_memory.neuron_groups[dg_name])
            )
            self.snn_memory.encode_pattern(dg_name, pattern, 1.0)
            
            # Store mapping
            self.neural_memory_map[memory_item['id']] = {
                'circuit': dg_name,
                'pattern': pattern,
                'type': memory_type
            }
            
        elif circuit_name.endswith("_semantic"):
            # Find available neurons and encode
            pattern = self._hash_to_binary_pattern(
                str(memory_item['content']), 
                50  # Use 50 neurons per semantic concept
            )
            
            # Find neurons to represent this memory
            ng = self.snn_memory.neuron_groups[circuit_name]
            for i in range(0, len(ng), 50):
                indices = list(range(i, min(i + 50, len(ng))))
                
                # Check if neurons are available
                if np.sum(ng.I[indices]) < 0.1:
                    # Encode pattern
                    self.snn_memory.set_input_current(
                        circuit_name,
                        indices[:len(pattern)],
                        pattern * memory_item['importance']
                    )
                    
                    # Store mapping
                    self.neural_memory_map[memory_item['id']] = {
                        'circuit': circuit_name,
                        'indices': indices,
                        'pattern': pattern,
                        'type': memory_type
                    }
                    break
        
    def _neural_encode_content(self, content: Any) -> np.ndarray:
        """
        Encode content for neural representation.
        
        Args:
            content: Content to encode
            
        Returns:
            Numerical encoding as numpy array
        """
        # For text content, encode using character values
        if isinstance(content, str):
            # Simple encoding: Use ASCII values normalized to [0,1]
            return np.array([ord(c)/255 for c in content[:50]])
            
        # For numerical content, use directly
        elif isinstance(content, (int, float)):
            return np.array([float(content)])
            
        # For lists/arrays, use directly (with limit)
        elif isinstance(content, (list, tuple, np.ndarray)):
            return np.array(content[:50])
            
        # For dictionaries, encode keys and values
        elif isinstance(content, dict):
            flattened = []
            for k, v in list(content.items())[:25]:
                flattened.append(hash(str(k)) % 255 / 255)
                if isinstance(v, (int, float)):
                    flattened.append(float(v) % 1.0)
                else:
                    flattened.append(hash(str(v)) % 255 / 255)
            return np.array(flattened)
            
        # For other types, use hash
        else:
            return np.array([hash(str(content)) % 255 / 255])
            
    def _hash_to_binary_pattern(self, text: str, size: int) -> np.ndarray:
        """
        Convert text to binary pattern using hash function.
        
        Args:
            text: Text to convert
            size: Size of binary pattern
            
        Returns:
            Binary pattern as numpy array
        """
        # Create a binary pattern based on hash of text
        hash_val = hash(text)
        
        # Convert hash to binary string
        binary = bin(abs(hash_val))[2:]
        
        # Ensure length by repeating if necessary
        while len(binary) < size:
            binary += binary
            
        # Truncate to exactly size
        binary = binary[:size]
        
        # Convert to numpy array
        return np.array([int(b) for b in binary])
        
    def consolidate_memory(self) -> None:
        """
        Consolidate memories from short-term to long-term memory.
        Simulates sleep-based memory consolidation process.
        """
        if not self.short_term_memory:
            return
            
        logger.info(f"Consolidating {len(self.short_term_memory)} memories")
        
        # Memories to consolidate
        important_memories = []
        
        # Sort by importance and recency
        stm_sorted = sorted(
            self.short_term_memory,
            key=lambda x: (x['importance'], x['timestamp']),
            reverse=True
        )
        
        # Select memories above importance threshold
        for memory in stm_sorted:
            if memory['importance'] >= 0.3:  # Threshold for consolidation
                important_memories.append(memory)
                
        # Process with quantum system if enabled
        if self.quantum_memory is not None:
            # Create quantum entanglement between related memories
            self._quantum_consolidate(important_memories)
                
        # Process with neural system if enabled
        if self.snn_memory is not None:
            # Simulate hippocampal replay during consolidation
            self._neural_consolidate(important_memories)
                
        # Add to long-term memory
        for memory in important_memories:
            # Update memory item
            memory['consolidated'] = True
            memory['consolidation_time'] = datetime.now().isoformat()
            
            # Move to appropriate memory system based on content type
            if 'type' in memory['metadata']:
                # Episodic memories (events, experiences)
                if memory['metadata']['type'] == 'episodic':
                    self.episodic_memory.append(memory)
                    logger.debug(f"Consolidated {memory['id']} to episodic memory")
                    
                    # Limit size of episodic memory
                    if len(self.episodic_memory) > self.episodic_capacity:
                        removed = self.episodic_memory.pop(0)
                        logger.debug(f"Episodic capacity reached, removed: {removed['id']}")
                    
                # Semantic memories (facts, concepts)
                elif memory['metadata']['type'] == 'semantic':
                    # Use content as key if it's a simple type
                    if isinstance(memory['content'], (str, int, float, bool)):
                        key = str(memory['content'])
                    else:
                        key = memory['id']
                        
                    self.semantic_memory[key] = memory
                    logger.debug(f"Consolidated {memory['id']} to semantic memory")
                    
                # Procedural memories (skills, procedures)
                elif memory['metadata']['type'] == 'procedural':
                    if 'skill' in memory['metadata']:
                        self.procedural_memory[memory['metadata']['skill']] = memory
                        logger.debug(f"Consolidated {memory['id']} to procedural memory")
            
            # Default to long-term memory
            else:
                self.long_term_memory.append(memory)
                logger.debug(f"Consolidated {memory['id']} to long-term memory")
                
                # Limit size of long-term memory
                if len(self.long_term_memory) > self.ltm_capacity:
                    # Remove least important memory
                    self.long_term_memory.sort(key=lambda x: x['importance'])
                    removed = self.long_term_memory.pop(0)
                    logger.debug(f"LTM capacity reached, removed: {removed['id']}")
                    
        # Clear short-term memory
        self.short_term_memory = []
        
        logger.info(f"Memory consolidation complete, consolidated {len(important_memories)} memories")
        
    def _quantum_consolidate(self, memories: List[Dict[str, Any]]) -> None:
        """
        Perform quantum memory consolidation.
        
        Args:
            memories: List of memory items to consolidate
        """
        # Get quantum IDs for memories
        quantum_ids = [
            self.quantum_memory_map[memory['id']]
            for memory in memories
            if memory['id'] in self.quantum_memory_map
        ]
        
        if len(quantum_ids) < 2:
            return  # Need at least 2 memories to entangle
            
        # Create entangled memory from these memories
        entangled_id = self.quantum_memory.create_entangled_memories(quantum_ids)
        
        # Find similar memories to create associations
        for i, memory1 in enumerate(memories[:-1]):
            for memory2 in memories[i+1:]:
                # Calculate similarity (could be more sophisticated)
                similarity = self._calculate_similarity(memory1['content'], memory2['content'])
                
                if similarity > 0.5:  # Threshold for association
                    # Create bidirectional association
                    if memory1['id'] not in self.associations:
                        self.associations[memory1['id']] = set()
                    if memory2['id'] not in self.associations:
                        self.associations[memory2['id']] = set()
                        
                    self.associations[memory1['id']].add(memory2['id'])
                    self.associations[memory2['id']].add(memory1['id'])
                    
                    logger.debug(f"Created memory association between {memory1['id']} and {memory2['id']}")
                    
        # Apply random memory decoherence to simulate forgetting
        self.quantum_memory.simulate_decoherence()
        
    def _neural_consolidate(self, memories: List[Dict[str, Any]]) -> None:
        """
        Perform neural memory consolidation with replay.
        
        Args:
            memories: List of memory items to consolidate
        """
        # Only process memories with neural encodings
        encoded_memories = [
            memory for memory in memories
            if memory['id'] in self.neural_memory_map
        ]
        
        if not encoded_memories:
            return
            
        # Simulate memory replay during sleep for each memory type
        working_memories = []
        episodic_memories = []
        semantic_memories = []
        
        # Group by type
        for memory in encoded_memories:
            info = self.neural_memory_map[memory['id']]
            
            if info['type'] == 'stm' or info['type'] == 'working':
                working_memories.append(memory)
            elif info['type'] == 'episodic':
                episodic_memories.append(memory)
            elif info['type'] == 'semantic' or info['type'] == 'ltm':
                semantic_memories.append(memory)
                
        # Process each type
        if working_memories:
            self._replay_working_memories(working_memories)
            
        if episodic_memories:
            self._replay_episodic_memories(episodic_memories)
            
        if semantic_memories:
            self._integrate_semantic_memories(semantic_memories)
            
    def _replay_working_memories(self, memories: List[Dict[str, Any]]) -> None:
        """
        Replay working memories for consolidation.
        
        Args:
            memories: Working memories to replay
        """
        circuit_name = f"{self.name}_working"
        
        # For each memory, replay neural activity
        for memory in memories:
            info = self.neural_memory_map[memory['id']]
            
            if 'indices' in info:
                # Activate neurons corresponding to this memory
                self.snn_memory.set_input_current(
                    circuit_name,
                    info['indices'],
                    np.ones(len(info['indices'])) * memory['importance']
                )
                
                # Run a short simulation to generate activity
                self.snn_memory.run_simulation(50)  # 50ms
                
                # Record spike pattern for transfer to episodic/semantic memory
                pattern = self.snn_memory.decode_pattern(circuit_name)
                
                # Transfer to episodic memory
                dg_name = f"{self.name}_episodic_DG"
                self.snn_memory.encode_pattern(dg_name, pattern)
                
                # Run simulation to encode in CA3
                self.snn_memory.run_simulation(20)  # 20ms
                
                # Update neural mapping to episodic
                self.neural_memory_map[memory['id']] = {
                    'circuit': dg_name,
                    'pattern': pattern,
                    'type': 'episodic'
                }
                
    def _replay_episodic_memories(self, memories: List[Dict[str, Any]]) -> None:
        """
        Replay episodic memories for consolidation.
        
        Args:
            memories: Episodic memories to replay
        """
        # For each memory, replay in hippocampal circuit
        for memory in memories:
            info = self.neural_memory_map[memory['id']]
            
            if 'pattern' in info:
                # Encode in DG layer
                dg_name = f"{self.name}_episodic_DG"
                self.snn_memory.encode_pattern(dg_name, info['pattern'])
                
                # Run simulation to activate CA3 and CA1
                self.snn_memory.run_simulation(30)  # 30ms
                
                # For high importance memories, also transfer to semantic memory
                if memory['importance'] > 0.7:
                    # Get activated pattern from CA1
                    ca1_name = f"{self.name}_episodic_CA1"
                    ca1_pattern = self.snn_memory.decode_pattern(ca1_name)
                    
                    # Encode in semantic memory
                    sem_name = f"{self.name}_semantic_excitatory"
                    
                    # Find available neurons
                    ng = self.snn_memory.neuron_groups[sem_name]
                    for i in range(0, len(ng), 50):
                        indices = list(range(i, min(i + 50, len(ng))))
                        
                        # Check if neurons are available
                        if np.sum(ng.I[indices]) < 0.1:
                            # Encode pattern
                            self.snn_memory.set_input_current(
                                sem_name,
                                indices[:len(ca1_pattern)],
                                ca1_pattern * memory['importance']
                            )
                            
                            # Run simulation to encode in semantic network
                            self.snn_memory.run_simulation(20)  # 20ms
                            
                            # Also update neural mapping
                            self.neural_memory_map[memory['id']] = {
                                'circuit': sem_name,
                                'indices': indices,
                                'pattern': ca1_pattern,
                                'type': 'semantic'
                            }
                            break
                            
    def _integrate_semantic_memories(self, memories: List[Dict[str, Any]]) -> None:
        """
        Integrate semantic memories for consolidation.
        
        Args:
            memories: Semantic memories to integrate
        """
        sem_name = f"{self.name}_semantic_excitatory"
        
        # Activate all semantic memories to strengthen connections
        for memory in memories:
            info = self.neural_memory_map[memory['id']]
            
            if 'indices' in info and 'pattern' in info:
                # Activate this semantic representation
                self.snn_memory.set_input_current(
                    sem_name,
                    info['indices'],
                    info['pattern'] * memory['importance']
                )
                
        # Run longer simulation to strengthen connections
        self.snn_memory.run_simulation(100)  # 100ms
        
    def _calculate_similarity(self, content1: Any, content2: Any) -> float:
        """
        Calculate similarity between two memory contents.
        
        Args:
            content1: First content
            content2: Second content
            
        Returns:
            Similarity score between 0 and 1
        """
        # Convert both to strings for simple comparison
        str1 = str(content1)
        str2 = str(content2)
        
        # Simple Jaccard similarity for strings
        set1 = set(str1.split() if isinstance(str1, str) else [str1])
        set2 = set(str2.split() if isinstance(str2, str) else [str2])
        
        if not set1 or not set2:
            return 0.0
            
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
        
    def replay_experiences(self) -> None:
        """
        Replay memories from replay buffer for reinforce learning.
        """
        if not self.replay_buffer:
            return
            
        logger.info(f"Replaying {len(self.replay_buffer)} experiences")
        
        # Sort by importance for prioritized replay
        self.replay_buffer.sort(key=lambda x: x['importance'], reverse=True)
        
        # Process with neural system if enabled
        if self.snn_memory is not None:
            # Replay top memories
            top_memories = self.replay_buffer[:min(10, len(self.replay_buffer))]
            
            # For each memory, replay neural activation
            for memory in top_memories:
                if memory['id'] in self.neural_memory_map:
                    info = self.neural_memory_map[memory['id']]
                    
                    if 'indices' in info:
                        # Reactivate neurons
                        self.snn_memory.set_input_current(
                            info['circuit'],
                            info['indices'],
                            np.ones(len(info['indices'])) * memory['importance']
                        )
                        
                        # Run a short simulation
                        self.snn_memory.run_simulation(30)  # 30ms
                        
                    elif 'pattern' in info:
                        # Reactivate pattern
                        self.snn_memory.encode_pattern(
                            info['circuit'],
                            info['pattern'],
                            memory['importance']
                        )
                        
                        # Run a short simulation
                        self.snn_memory.run_simulation(30)  # 30ms
                        
        # Process with quantum system if enabled
        if self.quantum_memory is not None:
            # Create superposition of replay memories
            quantum_ids = [
                self.quantum_memory_map[memory['id']]
                for memory in self.replay_buffer
                if memory['id'] in self.quantum_memory_map
            ]
            
            if quantum_ids:
                # Create quantum superposition
                weights = [
                    memory['importance'] 
                    for memory in self.replay_buffer
                    if memory['id'] in self.quantum_memory_map
                ]
                
                # Normalize weights for superposition
                sum_weights = sum(weights)
                if sum_weights > 0:
                    normalized_weights = [w / sum_weights for w in weights]
                    coefficients = [complex(w, 0) for w in normalized_weights]
                    
                    # Create superposition
                    self.quantum_memory.quantum_superposition_of_memories(
                        quantum_ids[:8],  # Limit to 8 memories for efficiency
                        coefficients[:8]
                    )
                    
        # Clear replay buffer
        self.replay_buffer = []
        
    def recall_memory(self, 
                     query: Any, 
                     memory_type: str = "all",
                     max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Recall memories matching a query.
        
        Args:
            query: Query to match against memories
            memory_type: Type of memory to search ("stm", "ltm", "episodic", "semantic", "all")
            max_results: Maximum number of results to return
            
        Returns:
            List of memory items matching the query
        """
        results = []
        
        # Convert query to string for comparison
        query_str = str(query)
        
        # Search in specified memory types
        if memory_type in ["stm", "all"]:
            for memory in self.short_term_memory:
                similarity = self._calculate_similarity(memory['content'], query)
                if similarity > 0.3:  # Threshold for matching
                    results.append({**memory, 'similarity': similarity})
                    
        if memory_type in ["ltm", "all"]:
            for memory in self.long_term_memory:
                similarity = self._calculate_similarity(memory['content'], query)
                if similarity > 0.3:
                    results.append({**memory, 'similarity': similarity})
                    
        if memory_type in ["episodic", "all"]:
            for memory in self.episodic_memory:
                similarity = self._calculate_similarity(memory['content'], query)
                if similarity > 0.3:
                    results.append({**memory, 'similarity': similarity})
                    
        if memory_type in ["semantic", "all"]:
            for key, memory in self.semantic_memory.items():
                similarity = self._calculate_similarity(memory['content'], query)
                if similarity > 0.3:
                    results.append({**memory, 'similarity': similarity})
                    
        # Search using quantum memory if enabled
        if self.quantum_memory is not None and memory_type in ["quantum", "all"]:
            quantum_results = self._quantum_recall(query)
            results.extend(quantum_results)
            
        # Search using neural memory if enabled
        if self.snn_memory is not None and memory_type in ["neural", "all"]:
            neural_results = self._neural_recall(query)
            results.extend(neural_results)
            
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Update access statistics for retrieved memories
        for memory in results[:max_results]:
            if 'id' in memory:
                self._update_memory_access(memory['id'])
                
        return results[:max_results]
        
    def _quantum_recall(self, query: Any) -> List[Dict[str, Any]]:
        """
        Recall memories using quantum memory system.
        
        Args:
            query: Query to match against memories
            
        Returns:
            List of memory items matching the query
        """
        results = []
        
        # Encode query as quantum state
        if isinstance(query, str):
            encoded = np.array([complex(ord(c) / 255, 0) for c in query])
        else:
            encoded = np.array([complex(float(hash(str(query)) % 255) / 255, 0)])
            
        # Ensure correct dimension
        if len(encoded) < self.quantum_memory.dimension:
            encoded = np.pad(encoded, (0, self.quantum_memory.dimension - len(encoded)))
        elif len(encoded) > self.quantum_memory.dimension:
            encoded = encoded[:self.quantum_memory.dimension]
            
        # Normalize
        encoded = encoded / np.linalg.norm(encoded)
        
        # Create a density matrix query state
        query_state = np.outer(encoded, encoded.conj())
        
        # Find similar states in quantum memory
        for memory_id, quantum_id in self.quantum_memory_map.items():
            # Retrieve quantum state
            state = self.quantum_memory.retrieve_memory(quantum_id)
            
            if state is not None:
                # Calculate fidelity (quantum similarity)
                fidelity = self.quantum_memory.operator.fidelity(query_state, state)
                
                if fidelity > 0.3:  # Threshold for matching
                    # Find original memory item
                    memory_item = None
                    
                    # Search in all memory systems
                    for memory in self.short_term_memory:
                        if memory['id'] == memory_id:
                            memory_item = memory
                            break
                            
                    if memory_item is None:
                        for memory in self.long_term_memory:
                            if memory['id'] == memory_id:
                                memory_item = memory
                                break
                                
                    if memory_item is None:
                        for memory in self.episodic_memory:
                            if memory['id'] == memory_id:
                                memory_item = memory
                                break
                                
                    if memory_item is None:
                        for key, memory in self.semantic_memory.items():
                            if memory['id'] == memory_id:
                                memory_item = memory
                                break
                                
                    if memory_item is not None:
                        results.append({**memory_item, 'similarity': fidelity})
                        
        return results
        
    def _neural_recall(self, query: Any) -> List[Dict[str, Any]]:
        """
        Recall memories using neural memory system.
        
        Args:
            query: Query to match against memories
            
        Returns:
            List of memory items matching the query
        """
        results = []
        
        # Encode query for neural representation
        query_encoding = self._neural_encode_content(query)
        
        # Create binary pattern
        query_pattern = (query_encoding > 0.5).astype(int)
        
        # Try to recall from each memory system
        # First, try semantic memory
        sem_name = f"{self.name}_semantic_excitatory"
        
        # Find neurons to represent this query
        ng = self.snn_memory.neuron_groups[sem_name]
        query_indices = list(range(min(50, len(ng))))
        
        # Encode query pattern
        self.snn_memory.set_input_current(
            sem_name,
            query_indices[:len(query_pattern)],
            query_pattern
        )
        
        # Run simulation to activate associated memories
        self.snn_memory.run_simulation(50)  # 50ms
        
        # Get activation pattern
        activation = self.snn_memory.decode_pattern(sem_name)
        
        # Compare with stored patterns
        for memory_id, info in self.neural_memory_map.items():
            if info['type'] == 'semantic' and 'pattern' in info:
                # Check pattern similarity
                pattern = info['pattern']
                overlap = np.sum(pattern * activation) / np.sum(pattern)
                
                if overlap > 0.4:  # Threshold for matching
                    # Find original memory item
                    memory_item = None
                    
                    # Search in all memory systems
                    for memory in self.short_term_memory:
                        if memory['id'] == memory_id:
                            memory_item = memory
                            break
                            
                    if memory_item is None:
                        for memory in self.long_term_memory:
                            if memory['id'] == memory_id:
                                memory_item = memory
                                break
                                
                    if memory_item is None:
                        for memory in self.episodic_memory:
                            if memory['id'] == memory_id:
                                memory_item = memory
                                break
                                
                    if memory_item is None:
                        for key, memory in self.semantic_memory.items():
                            if memory['id'] == memory_id:
                                memory_item = memory
                                break
                                
                    if memory_item is not None:
                        results.append({**memory_item, 'similarity': overlap})
                        
        # Also try episodic memory if semantic didn't yield enough results
        if len(results) < 3:
            # Try episodic memory
            dg_name = f"{self.name}_episodic_DG"
            
            # Create pattern for DG
            pattern_size = len(self.snn_memory.neuron_groups[dg_name])
            dg_pattern = self._hash_to_binary_pattern(str(query), pattern_size)
            
            # Encode pattern
            self.snn_memory.encode_pattern(dg_name, dg_pattern)
            
            # Run simulation to activate CA3 (pattern completion)
            self.snn_memory.run_simulation(30)  # 30ms
            
            # Get CA3 activation
            ca3_name = f"{self.name}_episodic_CA3"
            ca3_pattern = self.snn_memory.decode_pattern(ca3_name)
            
            # Compare with stored patterns
            for memory_id, info in self.neural_memory_map.items():
                if info['type'] == 'episodic' and 'pattern' in info:
                    # Check pattern similarity
                    pattern = info['pattern']
                    
                    # Ensure patterns are comparable in size
                    min_size = min(len(pattern), len(ca3_pattern))
                    pattern = pattern[:min_size]
                    ca3_pattern = ca3_pattern[:min_size]
                    
                    overlap = np.sum(pattern * ca3_pattern) / np.sum(pattern)
                    
                    if overlap > 0.4:  # Threshold for matching
                        # Find original memory item
                        memory_item = None
                        
                        # Search in all memory systems
                        for memory in self.episodic_memory:
                            if memory['id'] == memory_id:
                                memory_item = memory
                                break
                                
                        if memory_item is not None:
                            results.append({**memory_item, 'similarity': overlap})
                            
        return results
        
    def _update_memory_access(self, memory_id: str) -> None:
        """
        Update access statistics for a memory item.
        
        Args:
            memory_id: ID of memory to update
        """
        # Update in short-term memory
        for memory in self.short_term_memory:
            if memory['id'] == memory_id:
                memory['access_count'] += 1
                memory['last_accessed'] = datetime.now().isoformat()
                memory['importance'] = min(1.0, memory['importance'] + 0.1)  # Increase importance
                return
                
        # Update in long-term memory
        for memory in self.long_term_memory:
            if memory['id'] == memory_id:
                memory['access_count'] += 1
                memory['last_accessed'] = datetime.now().isoformat()
                memory['importance'] = min(1.0, memory['importance'] + 0.1)
                return
                
        # Update in episodic memory
        for memory in self.episodic_memory:
            if memory['id'] == memory_id:
                memory['access_count'] += 1
                memory['last_accessed'] = datetime.now().isoformat()
                memory['importance'] = min(1.0, memory['importance'] + 0.1)
                return
                
        # Update in semantic memory
        for key, memory in self.semantic_memory.items():
            if memory['id'] == memory_id:
                memory['access_count'] += 1
                memory['last_accessed'] = datetime.now().isoformat()
                memory['importance'] = min(1.0, memory['importance'] + 0.1)
                return
                
    def get_associated_memories(self, memory_id: str) -> List[Dict[str, Any]]:
        """
        Get memories associated with a given memory.
        
        Args:
            memory_id: ID of memory to get associations for
            
        Returns:
            List of associated memory items
        """
        if memory_id not in self.associations:
            return []
            
        associated_ids = self.associations[memory_id]
        associated_memories = []
        
        # Find associated memories in all memory systems
        for assoc_id in associated_ids:
            # Search in all memory systems
            for memory in self.short_term_memory:
                if memory['id'] == assoc_id:
                    associated_memories.append(memory)
                    break
                    
            if not any(mem['id'] == assoc_id for mem in associated_memories):
                for memory in self.long_term_memory:
                    if memory['id'] == assoc_id:
                        associated_memories.append(memory)
                        break
                        
            if not any(mem['id'] == assoc_id for mem in associated_memories):
                for memory in self.episodic_memory:
                    if memory['id'] == assoc_id:
                        associated_memories.append(memory)
                        break
                        
            if not any(mem['id'] == assoc_id for mem in associated_memories):
                for key, memory in self.semantic_memory.items():
                    if memory['id'] == assoc_id:
                        associated_memories.append(memory)
                        break
                        
        return associated_memories
        
    def save_memory_state(self, filename: str) -> None:
        """
        Save the current memory state to a file.
        
        Args:
            filename: Filename to save to
        """
        # Prepare memory state (excluding neural/quantum components)
        memory_state = {
            'id': self.id,
            'name': self.name,
            'short_term_memory': self.short_term_memory,
            'long_term_memory': self.long_term_memory,
            'episodic_memory': self.episodic_memory,
            'semantic_memory': self.semantic_memory,
            'procedural_memory': self.procedural_memory,
            'working_memory': self.working_memory,
            'associations': {k: list(v) for k, v in self.associations.items()},
            'stm_capacity': self.stm_capacity,
            'ltm_capacity': self.ltm_capacity,
            'episodic_capacity': self.episodic_capacity,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to file
        with open(filename, 'wb') as f:
            pickle.dump(memory_state, f)
            
        logger.info(f"Memory state saved to {filename}")
        
    def load_memory_state(self, filename: str) -> None:
        """
        Load memory state from a file.
        
        Args:
            filename: Filename to load from
        """
        # Load from file
        with open(filename, 'rb') as f:
            memory_state = pickle.load(f)
            
        # Restore memory state
        self.id = memory_state['id']
        self.name = memory_state['name']
        self.short_term_memory = memory_state['short_term_memory']
        self.long_term_memory = memory_state['long_term_memory']
        self.episodic_memory = memory_state['episodic_memory']
        self.semantic_memory = memory_state['semantic_memory']
        self.procedural_memory = memory_state['procedural_memory']
        self.working_memory = memory_state['working_memory']
        self.associations = {k: set(v) for k, v in memory_state['associations'].items()}
        self.stm_capacity = memory_state['stm_capacity']
        self.ltm_capacity = memory_state['ltm_capacity']
        self.episodic_capacity = memory_state['episodic_capacity']
        
        logger.info(f"Memory state loaded from {filename}")
        
        # Rebuild neural and quantum representations if needed
        if self.snn_memory is not None:
            self._rebuild_neural_representations()
            
        if self.quantum_memory is not None:
            self._rebuild_quantum_representations()
            
    def _rebuild_neural_representations(self) -> None:
        """Rebuild neural representations from symbolic memory."""
        # Clear existing neural memory maps
        self.neural_memory_map = {}
        
        # Rebuild for all memory systems
        for memory in self.short_term_memory:
            self._encode_in_neural_memory(memory, "stm")
            
        for memory in self.long_term_memory:
            self._encode_in_neural_memory(memory, "ltm")
            
        for memory in self.episodic_memory:
            self._encode_in_neural_memory(memory, "episodic")
            
        for key, memory in self.semantic_memory.items():
            self._encode_in_neural_memory(memory, "semantic")
            
    def _rebuild_quantum_representations(self) -> None:
        """Rebuild quantum representations from symbolic memory."""
        # Clear existing quantum memory maps
        self.quantum_memory_map = {}
        
        # Rebuild for all memory systems
        for memory in self.short_term_memory:
            self._store_in_quantum_memory(memory)
            
        for memory in self.long_term_memory:
            self._store_in_quantum_memory(memory)
            
        for memory in self.episodic_memory:
            self._store_in_quantum_memory(memory)
            
        for key, memory in self.semantic_memory.items():
            self._store_in_quantum_memory(memory)

"""
Consciousness Detection and Simulation Module

This module implements detection and measurement of consciousness-like behavior
in the Legion AGI system, based on theories of consciousness including
Global Workspace Theory, Integrated Information Theory, and quantum approaches.
"""

import numpy as np
import scipy.linalg
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from loguru import logger

from legion_agi.core.quantum_memory import QuantumMemory, CoherenceMeasures
from legion_agi.core.global_workspace import GlobalWorkspace, InformationUnit
from legion_agi.config import (
    GW_CONSCIOUSNESS_THRESHOLD,
    COHERENCE_THRESHOLD,
    NUM_QUBITS
)


class IntegratedInformationCalculator:
    """
    Calculates Integrated Information (Phi) based on Integrated Information Theory.
    This is a simplified implementation of ITT concepts.
    """
    
    def __init__(self, num_nodes: int):
        """
        Initialize Integrated Information calculator.
        
        Args:
            num_nodes: Number of nodes in the network
        """
        self.num_nodes = num_nodes
        
    def calculate_phi(self, 
                     transition_matrix: np.ndarray, 
                     current_state: np.ndarray) -> Dict[str, Any]:
        """
        Calculate integrated information (Phi) for a network.
        
        Args:
            transition_matrix: Transition probability matrix (NxN)
            current_state: Current state probability distribution
            
        Returns:
            Dictionary with Phi value and related metrics
        """
        if transition_matrix.shape != (self.num_nodes, self.num_nodes):
            raise ValueError(f"Transition matrix must be {self.num_nodes}x{self.num_nodes}")
            
        if current_state.shape != (self.num_nodes,):
            raise ValueError(f"Current state must have length {self.num_nodes}")
            
        # Calculate information content of the whole system
        whole_info = self._calculate_information(transition_matrix, current_state)
        
        # Try different partitions and find minimum information difference
        min_phi = float('inf')
        min_partition = None
        
        # For simplicity, we only consider bipartitions
        # For a complete calculation, all possible partitions should be considered
        for i in range(1, 2**(self.num_nodes-1)):
            # Create partition
            partition = self._int_to_partition(i, self.num_nodes)
            
            # Calculate information for partitioned system
            partitioned_info = self._calculate_partitioned_information(
                transition_matrix, current_state, partition
            )
            
            # Calculate Phi for this partition
            phi = whole_info - partitioned_info
            
            if phi < min_phi:
                min_phi = phi
                min_partition = partition
                
        # Ensure non-negative Phi (theoretical minimum is 0)
        min_phi = max(0, min_phi)
        
        return {
            "phi": min_phi,
            "min_partition": min_partition,
            "whole_information": whole_info
        }
        
    def _int_to_partition(self, n: int, size: int) -> List[int]:
        """
        Convert integer to binary partition.
        
        Args:
            n: Integer representation of partition
            size: System size
            
        Returns:
            Binary partition list (0s and 1s)
        """
        binary = bin(n)[2:].zfill(size)
        return [int(b) for b in binary]
        
    def _calculate_information(self, 
                              transition_matrix: np.ndarray, 
                              state: np.ndarray) -> float:
        """
        Calculate information content of a system.
        
        Args:
            transition_matrix: Transition matrix
            state: State probability distribution
            
        Returns:
            Information content
        """
        # Calculate entropy of current state
        entropy = self._calculate_entropy(state)
        
        # Calculate conditional entropy
        conditional_entropy = 0
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if transition_matrix[i, j] > 0 and state[i] > 0:
                    p_joint = transition_matrix[i, j] * state[i]
                    conditional_entropy -= p_joint * np.log2(transition_matrix[i, j])
                    
        # Mutual information
        information = entropy - conditional_entropy
        return information
        
    def _calculate_partitioned_information(self,
                                          transition_matrix: np.ndarray,
                                          state: np.ndarray,
                                          partition: List[int]) -> float:
        """
        Calculate information content of a partitioned system.
        
        Args:
            transition_matrix: Transition matrix
            state: State probability distribution
            partition: Partition definition (0s and 1s)
            
        Returns:
            Information content of the partitioned system
        """
        # Create partition indices
        part1 = [i for i, p in enumerate(partition) if p == 0]
        part2 = [i for i, p in enumerate(partition) if p == 1]
        
        if not part1 or not part2:
            return 0  # Trivial partition
            
        # Create reduced transition matrices
        trans_1 = self._reduce_transition_matrix(transition_matrix, part1)
        trans_2 = self._reduce_transition_matrix(transition_matrix, part2)
        
        # Create reduced states
        state_1 = state[part1]
        state_1 = state_1 / np.sum(state_1)  # Normalize
        
        state_2 = state[part2]
        state_2 = state_2 / np.sum(state_2)  # Normalize
        
        # Calculate information for each part
        info_1 = self._calculate_information(trans_1, state_1)
        info_2 = self._calculate_information(trans_2, state_2)
        
        # Total information of the partitioned system
        return info_1 + info_2
        
    def _reduce_transition_matrix(self, 
                                 matrix: np.ndarray, 
                                 indices: List[int]) -> np.ndarray:
        """
        Reduce transition matrix to specified indices.
        
        Args:
            matrix: Full transition matrix
            indices: Indices to keep
            
        Returns:
            Reduced transition matrix
        """
        # Extract submatrix
        reduced = matrix[np.ix_(indices, indices)]
        
        # Normalize rows to ensure probability distribution
        row_sums = np.sum(reduced, axis=1)
        for i in range(len(reduced)):
            if row_sums[i] > 0:
                reduced[i, :] /= row_sums[i]
                
        return reduced
        
    def _calculate_entropy(self, distribution: np.ndarray) -> float:
        """
        Calculate Shannon entropy of a probability distribution.
        
        Args:
            distribution: Probability distribution
            
        Returns:
            Shannon entropy
        """
        # Filter out zeros to avoid log(0)
        probs = distribution[distribution > 0]
        return -np.sum(probs * np.log2(probs))


class ConsciousnessDetector:
    """
    Detects and measures consciousness-like behavior in the system
    using multiple theories of consciousness.
    """
    
    def __init__(self, global_workspace: Optional[GlobalWorkspace] = None):
        """
        Initialize consciousness detector.
        
        Args:
            global_workspace: Global workspace instance
        """
        self.global_workspace = global_workspace
        self.quantum_memory = QuantumMemory(NUM_QUBITS)
        self.coherence_measures = CoherenceMeasures(2**NUM_QUBITS)
        self.iit_calculator = IntegratedInformationCalculator(num_nodes=10)  # Default size
        
    def detect_consciousness_gw(self) -> Dict[str, Any]:
        """
        Detect consciousness-like behavior based on Global Workspace Theory.
        
        Returns:
            Dictionary with consciousness metrics
        """
        if not self.global_workspace:
            return {"error": "Global workspace not available"}
            
        # Get current workspace contents
        contents = self.global_workspace.get_workspace_contents()
        
        # Get broadcast history
        history = self.global_workspace.get_broadcast_history()
        
        # Calculate metrics
        
        # 1. Current workspace occupancy
        occupancy = len(contents) / self.global_workspace.capacity if contents else 0
        
        # 2. Average activation of contents
        avg_activation = np.mean([info.activation for info in contents]) if contents else 0
        
        # 3. Broadcast intensity (recent broadcasts weighted by size)
        broadcast_intensity = 0
        if history:
            # Weight recent broadcasts more heavily
            for i, broadcast in enumerate(reversed(history[-10:])):
                weight = np.exp(-i * 0.2)  # Exponential decay with recency
                broadcast_intensity += weight * len(broadcast) / self.global_workspace.capacity
                
            broadcast_intensity /= min(len(history), 10)
            
        # 4. Diversity of sources
        source_diversity = 0
        if contents:
            sources = [info.source for info in contents]
            unique_sources = set(sources)
            source_diversity = len(unique_sources) / len(sources)
            
        # 5. Information persistence (how long information stays in workspace)
        persistence = 0
        if contents:
            avg_broadcast_count = np.mean([info.broadcast_count for info in contents])
            persistence = avg_broadcast_count / 3  # Normalize by typical broadcast cycles
            
        # Overall consciousness score based on GWT
        # Weighted combination of metrics
        consciousness_score = (
            0.3 * occupancy +
            0.2 * avg_activation +
            0.2 * broadcast_intensity +
            0.15 * source_diversity +
            0.15 * persistence
        )
        
        # Conscious access determination
        conscious_access = consciousness_score >= GW_CONSCIOUSNESS_THRESHOLD
        
        return {
            "consciousness_score": consciousness_score,
            "conscious_access": conscious_access,
            "metrics": {
                "workspace_occupancy": occupancy,
                "average_activation": avg_activation,
                "broadcast_intensity": broadcast_intensity,
                "source_diversity": source_diversity,
                "information_persistence": persistence
            }
        }
        
    def detect_consciousness_quantum(self, quantum_state: np.ndarray) -> Dict[str, Any]:
        """
        Detect consciousness-like behavior based on quantum theories of consciousness.
        
        Args:
            quantum_state: Quantum density matrix
            
        Returns:
            Dictionary with quantum consciousness metrics
        """
        # Store state in quantum memory
        self.quantum_memory.state = quantum_state
        
        # Calculate quantum coherence measures
        coherence_measures = self.quantum_memory.get_coherence_measures()
        
        # Normalize l1-norm coherence (maximum possible is dimension-1)
        dimension = quantum_state.shape[0]
        normalized_coherence = coherence_measures["l1_norm"] / (dimension - 1)
        
        # Calculate quantum metrics relevant to consciousness theories
        
        # 1. Quantum superposition degree (from eigenvalue analysis)
        eigenvalues = np.linalg.eigvalsh(quantum_state)
        superposition = 1 - np.max(eigenvalues)  # 1 - largest population
        
        # 2. Quantum entanglement (simplified measure)
        # For a more accurate measure, would need to identify subsystems
        if self.quantum_memory.num_qubits >= 2:
            # Use subsystem structure of first qubit vs rest
            entanglement = self.quantum_memory.calculate_entanglement_entropy([0])
            # Normalize by maximum entropy (log2 of subsystem dimension)
            normalized_entanglement = entanglement / 1.0  # log2(2) = 1 for first qubit
        else:
            normalized_entanglement = 0
            
        # 3. Quantum coherence relative to decoherence threshold
        coherence_ratio = normalized_coherence / COHERENCE_THRESHOLD
        
        # Overall quantum consciousness score
        quantum_consciousness = (
            0.4 * normalized_coherence +
            0.3 * superposition +
            0.3 * normalized_entanglement
        )
        
        # Consciousness present according to quantum criteria
        quantum_conscious = quantum_consciousness >= COHERENCE_THRESHOLD
        
        return {
            "quantum_consciousness_score": quantum_consciousness,
            "quantum_conscious": quantum_conscious,
            "metrics": {
                "normalized_coherence": normalized_coherence,
                "coherence_ratio": coherence_ratio,
                "superposition_degree": superposition,
                "normalized_entanglement": normalized_entanglement,
                "detailed_coherence": coherence_measures
            }
        }
        
    def detect_consciousness_iit(self, 
                                network_matrix: np.ndarray, 
                                current_state: np.ndarray) -> Dict[str, Any]:
        """
        Detect consciousness-like behavior based on Integrated Information Theory.
        
        Args:
            network_matrix: Connectivity/transition matrix of the network
            current_state: Current state of the network
            
        Returns:
            Dictionary with IIT consciousness metrics
        """
        # Resize calculator if needed
        if network_matrix.shape[0] != self.iit_calculator.num_nodes:
            self.iit_calculator = IntegratedInformationCalculator(network_matrix.shape[0])
            
        # Calculate Phi (integrated information)
        phi_result = self.iit_calculator.calculate_phi(network_matrix, current_state)
        
        # Maximum possible Phi depends on network size
        # For binary nodes, theoretical max is approximately log2(N)
        n = network_matrix.shape[0]
        theoretical_max_phi = np.log2(n)
        
        # Normalize Phi
        normalized_phi = phi_result["phi"] / theoretical_max_phi if theoretical_max_phi > 0 else 0
        
        # Minimum information integration for consciousness (arbitrary threshold)
        consciousness_threshold = 0.3  # 30% of theoretical maximum
        
        # Determine consciousness
        iit_conscious = normalized_phi >= consciousness_threshold
        
        return {
            "phi": phi_result["phi"],
            "normalized_phi": normalized_phi,
            "iit_conscious": iit_conscious,
            "min_partition": phi_result["min_partition"],
            "whole_information": phi_result["whole_information"],
            "theoretical_max_phi": theoretical_max_phi,
            "consciousness_threshold": consciousness_threshold
        }
        
    def detect_integrated_consciousness(self, 
                                      quantum_state: Optional[np.ndarray] = None,
                                      network_matrix: Optional[np.ndarray] = None,
                                      current_state: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Detect consciousness using multiple theories integrated together.
        
        Args:
            quantum_state: Quantum density matrix (for quantum theories)
            network_matrix: Network connectivity matrix (for IIT)
            current_state: Current network state (for IIT)
            
        Returns:
            Dictionary with integrated consciousness assessment
        """
        # Initialize results for each theory
        gw_results = None
        quantum_results = None
        iit_results = None
        
        # Get results from each available theory
        if self.global_workspace:
            gw_results = self.detect_consciousness_gw()
            
        if quantum_state is not None:
            quantum_results = self.detect_consciousness_quantum(quantum_state)
            
        if network_matrix is not None and current_state is not None:
            iit_results = self.detect_consciousness_iit(network_matrix, current_state)
            
        # Calculate integrated consciousness
        consciousness_indicators = []
        theory_weights = {}
        
        if gw_results and "consciousness_score" in gw_results:
            consciousness_indicators.append(gw_results["consciousness_score"])
            theory_weights["gw"] = 0.4  # Weight for GWT
            
        if quantum_results and "quantum_consciousness_score" in quantum_results:
            consciousness_indicators.append(quantum_results["quantum_consciousness_score"])
            theory_weights["quantum"] = 0.3  # Weight for quantum theories
            
        if iit_results and "normalized_phi" in iit_results:
            consciousness_indicators.append(iit_results["normalized_phi"])
            theory_weights["iit"] = 0.3  # Weight for IIT
            
        # Calculate integrated consciousness score
        if consciousness_indicators:
            # If weights sum to less than 1, normalize them
            weight_sum = sum(theory_weights.values())
            if weight_sum < 1.0:
                theory_weights = {k: v/weight_sum for k, v in theory_weights.items()}
                
            # Calculate weighted score
            integrated_score = 0
            i = 0
            
            if "gw" in theory_weights and gw_results:
                integrated_score += theory_weights["gw"] * consciousness_indicators[i]
                i += 1
                
            if "quantum" in theory_weights and quantum_results:
                integrated_score += theory_weights["quantum"] * consciousness_indicators[i]
                i += 1
                
            if "iit" in theory_weights and iit_results:
                integrated_score += theory_weights["iit"] * consciousness_indicators[i]
                
            # Determine overall consciousness
            # Using mean of thresholds as integrated threshold
            thresholds = []
            
            if gw_results:
                thresholds.append(GW_CONSCIOUSNESS_THRESHOLD)
                
            if quantum_results:
                thresholds.append(COHERENCE_THRESHOLD)
                
            if iit_results:
                thresholds.append(iit_results["consciousness_threshold"])
                
            integrated_threshold = np.mean(thresholds) if thresholds else 0.5
            integrated_conscious = integrated_score >= integrated_threshold
            
        else:
            integrated_score = 0
            integrated_threshold = 0.5
            integrated_conscious = False
            
        # Construct result
        result = {
            "integrated_consciousness_score": integrated_score,
            "integrated_conscious": integrated_conscious,
            "integrated_threshold": integrated_threshold,
            "theory_weights": theory_weights,
            "theories_available": {
                "global_workspace": gw_results is not None,
                "quantum": quantum_results is not None,
                "integrated_information": iit_results is not None
            }
        }
        
        # Include individual theory results if available
        if gw_results:
            result["global_workspace"] = gw_results
            
        if quantum_results:
            result["quantum"] = quantum_results
            
        if iit_results:
            result["integrated_information"] = iit_results
            
        return result
        
    def monitor_consciousness_emergence(self, 
                                      num_steps: int = 10, 
                                      step_duration: float = 1.0) -> Dict[str, Any]:
        """
        Monitor the emergence of consciousness-like behavior over time.
        
        Args:
            num_steps: Number of monitoring steps
            step_duration: Duration between steps
            
        Returns:
            Dictionary with consciousness emergence data
        """
        import time
        
        # Initialize monitoring data
        consciousness_scores = []
        timestamps = []
        consciousness_metrics = []
        
        for step in range(num_steps):
            # Record timestamp
            timestamps.append(time.time())
            
            # Detect current consciousness state
            result = None
            
            # Use available theories based on what's available in the system
            if self.global_workspace:
                # For GWT only
                result = self.detect_consciousness_gw()
                consciousness_scores.append(result["consciousness_score"])
                consciousness_metrics.append(result["metrics"])
            else:
                # No measurements available
                consciousness_scores.append(0)
                consciousness_metrics.append({})
                
            # Wait for next step
            if step < num_steps - 1:
                time.sleep(step_duration)
                
        # Calculate emergence metrics
        if consciousness_scores:
            # Rate of change in consciousness
            if len(consciousness_scores) > 1:
                emergence_rate = (consciousness_scores[-1] - consciousness_scores[0]) / (num_steps - 1)
            else:
                emergence_rate = 0
                
            # Variability in consciousness
            consciousness_variability = np.std(consciousness_scores) if len(consciousness_scores) > 1 else 0
            
            # Detect patterns in emergence (simple trend analysis)
            if len(consciousness_scores) > 2:
                # Calculate differences between consecutive scores
                diffs = np.diff(consciousness_scores)
                
                # Check if consistently increasing
                increasing = np.all(diffs > -0.01)  # Allow small decreases
                
                # Check if consistently decreasing
                decreasing = np.all(diffs < 0.01)  # Allow small increases
                
                # Check if fluctuating (alternating increases and decreases)
                alternating = np.all(diffs[:-1] * diffs[1:] < 0)
                
                # Determine pattern
                if increasing:
                    pattern = "consistently_increasing"
                elif decreasing:
                    pattern = "consistently_decreasing"
                elif alternating:
                    pattern = "fluctuating"
                else:
                    pattern = "irregular"
            else:
                pattern = "insufficient_data"
                
        else:
            emergence_rate = 0
            consciousness_variability = 0
            pattern = "no_data"
            
        return {
            "consciousness_scores": consciousness_scores,
            "timestamps": timestamps,
            "consciousness_metrics": consciousness_metrics,
            "emergence_rate": emergence_rate,
            "consciousness_variability": consciousness_variability,
            "emergence_pattern": pattern,
            "monitoring_duration": num_steps * step_duration
        }

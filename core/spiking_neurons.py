"""
Spiking Neural Network Simulation using Brian2

This module implements biologically-inspired spiking neural networks for
memory and cognitive functions in the Legion AGI system. It uses the Brian2
library to simulate spiking neuron dynamics.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from loguru import logger

from brian2 import (
    NeuronGroup, 
    Synapses, 
    SpikeMonitor, 
    StateMonitor, 
    start_scope, 
    run, 
    ms, 
    mV, 
    nA,
    Hz,
    second
)

from legion_agi.config import (
    SNN_SIMULATION_TIME,
    SNN_INTEGRATION_TIME,
    SNN_THRESHOLD,
    SNN_RESET,
    SNN_REFRACTORY_PERIOD
)


class SpikingNeuralNetwork:
    """
    Spiking Neural Network simulation using Brian2.
    Implements biologically plausible neural dynamics for memory and cognition.
    """
    
    def __init__(self):
        """Initialize Spiking Neural Network."""
        # Start Brian2 scope
        start_scope()
        
        # Network components
        self.neuron_groups = {}
        self.synapses = {}
        self.spike_monitors = {}
        self.state_monitors = {}
        
        # Default parameters
        self.simulation_time = SNN_SIMULATION_TIME * ms
        self.integration_time = SNN_INTEGRATION_TIME * ms
        self.threshold = SNN_THRESHOLD
        self.reset = SNN_RESET
        self.refractory_period = SNN_REFRACTORY_PERIOD * ms
        
        logger.info("Spiking Neural Network initialized")
        
    def create_neuron_group(self, 
                           name: str, 
                           num_neurons: int, 
                           neuron_model: str = 'LIF',
                           parameters: Optional[Dict[str, float]] = None) -> None:
        """
        Create a group of neurons with specified model.
        
        Args:
            name: Name of the neuron group
            num_neurons: Number of neurons in the group
            neuron_model: Model type ('LIF', 'AdEx', 'Izhikevich')
            parameters: Additional model parameters
        """
        params = parameters or {}
        
        if neuron_model == 'LIF':
            # Leaky Integrate-and-Fire model
            model_eqs = '''
            dv/dt = (I - v) / tau : 1
            I : 1
            tau : second
            '''
            
            # Default parameters
            tau = params.get('tau', 10 * ms)
            
            # Create neuron group
            ng = NeuronGroup(
                num_neurons,
                model_eqs,
                threshold=f'v > {self.threshold}',
                reset=f'v = {self.reset}',
                refractory=self.refractory_period,
                method='euler'
            )
            
            # Set parameters
            ng.tau = tau
            ng.v = params.get('v_init', 0)
            ng.I = params.get('I_init', 0)
            
        elif neuron_model == 'AdEx':
            # Adaptive Exponential Integrate-and-Fire model
            model_eqs = '''
            dv/dt = (-(v - EL) + delta_T * exp((v - VT) / delta_T) + R * I - w) / tau_m : volt
            dw/dt = (a * (v - EL) - w) / tau_w : amp
            I : amp
            '''
            
            # Default parameters
            tau_m = params.get('tau_m', 20 * ms)
            tau_w = params.get('tau_w', 200 * ms)
            EL = params.get('EL', -70 * mV)
            VT = params.get('VT', -50 * mV)
            delta_T = params.get('delta_T', 2 * mV)
            a = params.get('a', 4 * nA/mV)
            b = params.get('b', 80 * pA)
            R = params.get('R', 100 * Mohm)
            v_reset = params.get('v_reset', -55 * mV)
            
            # Create neuron group
            ng = NeuronGroup(
                num_neurons,
                model_eqs,
                threshold='v > -30*mV',
                reset=f'v = {v_reset}; w += {b}',
                refractory=self.refractory_period,
                method='euler'
            )
            
            # Set parameters
            ng.v = EL
            ng.w = 0 * amp
            ng.I = 0 * amp
            
        elif neuron_model == 'Izhikevich':
            # Izhikevich model
            model_eqs = '''
            dv/dt = (0.04 * v**2 + 5*v + 140 - u + I) / ms : 1
            du/dt = (a * (b*v - u)) / ms : 1
            I : 1
            a : 1
            b : 1
            c : 1
            d : 1
            '''
            
            # Default parameters (regular spiking)
            a = params.get('a', 0.02)
            b = params.get('b', 0.2)
            c = params.get('c', -65)
            d = params.get('d', 8)
            
            # Create neuron group
            ng = NeuronGroup(
                num_neurons,
                model_eqs,
                threshold='v >= 30',
                reset='v = c; u += d',
                method='euler'
            )
            
            # Set parameters
            ng.a = a
            ng.b = b
            ng.c = c
            ng.d = d
            ng.v = c
            ng.u = b * c
            ng.I = 0
            
        else:
            raise ValueError(f"Unknown neuron model: {neuron_model}")
            
        # Store neuron group
        self.neuron_groups[name] = ng
        
        # Create spike monitor
        self.spike_monitors[name] = SpikeMonitor(ng)
        
        logger.info(f"Created neuron group '{name}' with {num_neurons} {neuron_model} neurons")
        
    def create_synapse(self, 
                      name: str, 
                      source: str, 
                      target: str, 
                      connectivity: str = 'all',
                      synapse_type: str = 'static',
                      parameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Create synapses between neuron groups.
        
        Args:
            name: Name of the synapse
            source: Source neuron group name
            target: Target neuron group name
            connectivity: Connectivity pattern ('all', 'random', 'one-to-one')
            synapse_type: Type of synapse ('static', 'STDP', 'facilitating')
            parameters: Additional synapse parameters
        """
        params = parameters or {}
        
        # Get neuron groups
        source_group = self.neuron_groups.get(source)
        target_group = self.neuron_groups.get(target)
        
        if source_group is None or target_group is None:
            raise ValueError(f"Source or target neuron group not found: {source}, {target}")
            
        if synapse_type == 'static':
            # Static synapse model
            model_eqs = '''
            w : 1
            '''
            
            # Create synapses
            syn = Synapses(
                source_group,
                target_group,
                model=model_eqs,
                on_pre=f'I_post += w'
            )
            
        elif synapse_type == 'STDP':
            # Spike-timing dependent plasticity
            model_eqs = '''
            w : 1
            dapre/dt = -apre / tau_pre : 1 (event-driven)
            dapost/dt = -apost / tau_post : 1 (event-driven)
            '''
            
            # Default parameters
            tau_pre = params.get('tau_pre', 20 * ms)
            tau_post = params.get('tau_post', 20 * ms)
            A_pre = params.get('A_pre', 0.01)
            A_post = params.get('A_post', -0.01)
            w_max = params.get('w_max', 1.0)
            
            # Create synapses
            syn = Synapses(
                source_group,
                target_group,
                model=model_eqs,
                on_pre=f'''
                I_post += w
                apre += {A_pre}
                w = clip(w + apost, 0, {w_max})
                ''',
                on_post=f'''
                apost += {A_post}
                w = clip(w + apre, 0, {w_max})
                '''
            )
            
        elif synapse_type == 'facilitating':
            # Short-term plasticity (facilitating)
            model_eqs = '''
            w : 1
            u : 1
            x : 1
            du/dt = (U - u) / tau_f : 1 (event-driven)
            dx/dt = (1 - x) / tau_d : 1 (event-driven)
            '''
            
            # Default parameters
            U = params.get('U', 0.1)
            tau_f = params.get('tau_f', 100 * ms)
            tau_d = params.get('tau_d', 800 * ms)
            
            # Create synapses
            syn = Synapses(
                source_group,
                target_group,
                model=model_eqs,
                on_pre=f'''
                u = u + U * (1 - u)
                r = u * x
                x = x * (1 - u)
                I_post += w * r
                '''
            )
            
            # Set initial values
            syn.u = 0
            syn.x = 1
            
        else:
            raise ValueError(f"Unknown synapse type: {synapse_type}")
            
        # Set connectivity
        if connectivity == 'all':
            syn.connect()
        elif connectivity == 'random':
            p = params.get('p', 0.1)
            syn.connect(p=p)
        elif connectivity == 'one-to-one':
            syn.connect(j='i')
        else:
            raise ValueError(f"Unknown connectivity pattern: {connectivity}")
            
        # Set weights
        w = params.get('w', 0.5)
        syn.w = w
        
        # Store synapse
        self.synapses[name] = syn
        
        logger.info(f"Created {synapse_type} synapse '{name}' from '{source}' to '{target}' with {connectivity} connectivity")
        
    def create_neuronal_memory(self, 
                              name: str, 
                              memory_size: int,
                              inhibition_strength: float = 1.0) -> None:
        """
        Create a neuronal memory circuit (attractor network).
        
        Args:
            name: Base name for the memory components
            memory_size: Number of memory neurons
            inhibition_strength: Strength of lateral inhibition
        """
        # Create excitatory memory neurons
        self.create_neuron_group(
            f"{name}_excitatory",
            memory_size,
            'LIF',
            {'tau': 20 * ms}
        )
        
        # Create inhibitory interneurons
        self.create_neuron_group(
            f"{name}_inhibitory",
            memory_size // 4,  # Fewer inhibitory neurons (biological ratio)
            'LIF',
            {'tau': 10 * ms}  # Faster time constant for inhibitory neurons
        )
        
        # Create recurrent excitatory connections
        self.create_synapse(
            f"{name}_recurrent",
            f"{name}_excitatory",
            f"{name}_excitatory",
            'random',
            'STDP',
            {'p': 0.2, 'w': 0.5}
        )
        
        # Create excitatory -> inhibitory connections
        self.create_synapse(
            f"{name}_exc_to_inh",
            f"{name}_excitatory",
            f"{name}_inhibitory",
            'random',
            'static',
            {'p': 0.5, 'w': 0.2}
        )
        
        # Create inhibitory -> excitatory connections (lateral inhibition)
        self.create_synapse(
            f"{name}_inh_to_exc",
            f"{name}_inhibitory",
            f"{name}_excitatory",
            'random',
            'static',
            {'p': 0.5, 'w': -inhibition_strength}  # Negative weight for inhibition
        )
        
        logger.info(f"Created neuronal memory circuit '{name}' with {memory_size} memory neurons")
        
    def create_working_memory_circuit(self, 
                                    name: str,
                                    num_items: int = 7,  # Miller's Law (7±2)
                                    neurons_per_item: int = 50) -> None:
        """
        Create a working memory circuit with persistent activity.
        
        Args:
            name: Base name for the memory components
            num_items: Number of items that can be held in working memory
            neurons_per_item: Number of neurons per memory item
        """
        # Create stimulus input neurons
        self.create_neuron_group(
            f"{name}_input",
            num_items,
            'LIF',
            {'tau': 5 * ms}  # Fast response for inputs
        )
        
        # Create persistent memory neurons (with recurrent connections)
        self.create_neuron_group(
            f"{name}_persistent",
            num_items * neurons_per_item,
            'AdEx',  # AdEx model can exhibit persistent activity
            {
                'tau_m': 20 * ms,
                'tau_w': 600 * ms,  # Slow adaptation for persistence
                'a': 4 * nA/mV,
                'b': 20 * pA  # Lower b for increased excitability
            }
        )
        
        # Create inhibitory control neurons
        self.create_neuron_group(
            f"{name}_inhibitory",
            num_items,
            'LIF',
            {'tau': 10 * ms}
        )
        
        # Create input -> persistent connections
        self.create_synapse(
            f"{name}_input_to_persistent",
            f"{name}_input",
            f"{name}_persistent",
            'random',
            'facilitating',  # Facilitating synapses for working memory
            {
                'p': 0.7,
                'w': 1.0,
                'U': 0.05,  # Low U for facilitation
                'tau_f': 1000 * ms,  # Long facilitation time constant
                'tau_d': 200 * ms
            }
        )
        
        # Create recurrent connections within persistent neurons (for each item)
        for i in range(num_items):
            start_idx = i * neurons_per_item
            end_idx = (i + 1) * neurons_per_item
            
            # Define specific indexes for this subgroup
            neurons = self.neuron_groups[f"{name}_persistent"][start_idx:end_idx]
            
            # Create recurrent subgroup synapse
            syn = Synapses(
                neurons,
                neurons,
                model='''
                w : 1
                ''',
                on_pre='I_post += w'
            )
            
            # Connect with high probability within the subgroup
            syn.connect(p=0.5)
            syn.w = 0.2
            
            # Store in synapses dictionary
            self.synapses[f"{name}_persistent_recurrent_{i}"] = syn
            
        # Create persistent -> inhibitory connections
        self.create_synapse(
            f"{name}_persistent_to_inhibitory",
            f"{name}_persistent",
            f"{name}_inhibitory",
            'random',
            'static',
            {'p': 0.3, 'w': 0.5}
        )
        
        # Create inhibitory -> persistent connections
        self.create_synapse(
            f"{name}_inhibitory_to_persistent",
            f"{name}_inhibitory",
            f"{name}_persistent",
            'random',
            'static',
            {'p': 0.3, 'w': -1.0}  # Strong inhibition for gating
        )
        
        logger.info(f"Created working memory circuit '{name}' for {num_items} items with {neurons_per_item} neurons per item")
        
    def create_hippocampal_memory(self,
                                name: str,
                                num_patterns: int = 100,
                                pattern_size: int = 50) -> None:
        """
        Create a hippocampal-inspired memory system (CA3-like).
        
        Args:
            name: Base name for the memory components
            num_patterns: Number of patterns that can be stored
            pattern_size: Number of neurons per pattern
        """
        # Create CA3-like recurrent network (with high sparsity)
        self.create_neuron_group(
            f"{name}_CA3",
            num_patterns * pattern_size,
            'LIF',
            {'tau': 20 * ms}
        )
        
        # Create input layer (dentate gyrus-like)
        self.create_neuron_group(
            f"{name}_DG",
            num_patterns * 4,  # 4x pattern separation
            'LIF',
            {'tau': 10 * ms}
        )
        
        # Create output layer (CA1-like)
        self.create_neuron_group(
            f"{name}_CA1",
            num_patterns * pattern_size // 2,  # Compression
            'LIF',
            {'tau': 20 * ms}
        )
        
        # Create input -> CA3 connections (sparse, strong)
        self.create_synapse(
            f"{name}_DG_to_CA3",
            f"{name}_DG",
            f"{name}_CA3",
            'random',
            'STDP',
            {'p': 0.05, 'w': 0.9, 'w_max': 1.5}  # Sparse, strong connections
        )
        
        # Create recurrent CA3 connections (for pattern completion)
        self.create_synapse(
            f"{name}_CA3_recurrent",
            f"{name}_CA3",
            f"{name}_CA3",
            'random',
            'STDP',
            {'p': 0.15, 'w': 0.4, 'w_max': 1.0}  # Moderately sparse recurrent connections
        )
        
        # Create CA3 -> CA1 connections
        self.create_synapse(
            f"{name}_CA3_to_CA1",
            f"{name}_CA3",
            f"{name}_CA1",
            'random',
            'STDP',
            {'p': 0.2, 'w': 0.7, 'w_max': 1.2}
        )
        
        logger.info(f"Created hippocampal-inspired memory circuit '{name}' with {num_patterns} patterns")
        
    def set_input_current(self, group_name: str, neuron_indices: List[int], current: float) -> None:
        """
        Set input current to specific neurons in a group.
        
        Args:
            group_name: Name of the neuron group
            neuron_indices: Indices of neurons to stimulate
            current: Current value to set
        """
        if group_name not in self.neuron_groups:
            raise ValueError(f"Neuron group not found: {group_name}")
            
        ng = self.neuron_groups[group_name]
        ng.I[neuron_indices] = current
        
        logger.debug(f"Set input current {current} for {len(neuron_indices)} neurons in group '{group_name}'")
        
    def encode_pattern(self, group_name: str, pattern: np.ndarray, current: float = 1.0) -> None:
        """
        Encode a binary pattern as input currents.
        
        Args:
            group_name: Name of the neuron group
            pattern: Binary pattern (1s and 0s)
            current: Current value for active neurons (1s in pattern)
        """
        if group_name not in self.neuron_groups:
            raise ValueError(f"Neuron group not found: {group_name}")
            
        ng = self.neuron_groups[group_name]
        
        if len(pattern) > len(ng):
            raise ValueError(f"Pattern size ({len(pattern)}) exceeds neuron group size ({len(ng)})")
            
        # Reset all currents to 0
        ng.I = 0
        
        # Set currents for active neurons
        active_indices = np.where(pattern == 1)[0]
        ng.I[active_indices] = current
        
        logger.debug(f"Encoded pattern with {len(active_indices)} active neurons in group '{group_name}'")
        
    def encode_data(self, group_name: str, data: np.ndarray, max_current: float = 1.0) -> None:
        """
        Encode continuous data as input currents.
        
        Args:
            group_name: Name of the neuron group
            data: Continuous data array (will be normalized)
            max_current: Maximum current value after normalization
        """
        if group_name not in self.neuron_groups:
            raise ValueError(f"Neuron group not found: {group_name}")
            
        ng = self.neuron_groups[group_name]
        
        if len(data) > len(ng):
            raise ValueError(f"Data size ({len(data)}) exceeds neuron group size ({len(ng)})")
            
        # Normalize data to [0, max_current]
        if np.max(data) != np.min(data):
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data)) * max_current
        else:
            normalized_data = np.zeros_like(data)
            
        # Set currents
        ng.I[:len(data)] = normalized_data
        
        logger.debug(f"Encoded continuous data in group '{group_name}'")
        
    def run_simulation(self, duration: Optional[float] = None) -> None:
        """
        Run simulation for a specified duration.
        
        Args:
            duration: Simulation duration in ms (default: from config)
        """
        sim_time = duration * ms if duration is not None else self.simulation_time
        
        logger.info(f"Running simulation for {sim_time/ms} ms")
        run(sim_time)
        
    def get_spike_counts(self, group_name: str) -> np.ndarray:
        """
        Get spike counts for all neurons in a group.
        
        Args:
            group_name: Name of the neuron group
            
        Returns:
            Array of spike counts for each neuron
        """
        if group_name not in self.spike_monitors:
            raise ValueError(f"Spike monitor not found for group: {group_name}")
            
        monitor = self.spike_monitors[group_name]
        
        # Count spikes for each neuron
        unique, counts = np.unique(monitor.i, return_counts=True)
        
        # Create full array with zeros for neurons that didn't spike
        full_counts = np.zeros(len(self.neuron_groups[group_name]))
        full_counts[unique] = counts
        
        return full_counts
        
    def get_spike_trains(self, group_name: str) -> Dict[int, np.ndarray]:
        """
        Get spike times for all neurons in a group.
        
        Args:
            group_name: Name of the neuron group
            
        Returns:
            Dictionary mapping neuron indices to arrays of spike times
        """
        if group_name not in self.spike_monitors:
            raise ValueError(f"Spike monitor not found for group: {group_name}")
            
        monitor = self.spike_monitors[group_name]
        
        # Group spike times by neuron index
        spike_trains = {}
        for i in range(len(self.neuron_groups[group_name])):
            mask = monitor.i == i
            spike_trains[i] = monitor.t[mask] / ms  # Convert to ms
            
        return spike_trains
        
    def decode_pattern(self, group_name: str, threshold: float = 0.5) -> np.ndarray:
        """
        Decode spike activity as a binary pattern.
        
        Args:
            group_name: Name of the neuron group
            threshold: Spike count threshold for considering a neuron active
            
        Returns:
            Binary pattern representing neuron activity
        """
        spike_counts = self.get_spike_counts(group_name)
        
        # Normalize by maximum spike count
        if np.max(spike_counts) > 0:
            normalized_counts = spike_counts / np.max(spike_counts)
        else:
            normalized_counts = spike_counts
            
        # Apply threshold
        binary_pattern = (normalized_counts > threshold).astype(int)
        
        return binary_pattern
        
    def reset_network(self) -> None:
        """Reset the entire network (clear all activity)."""
        # Start a new scope (clears all objects)
        start_scope()
        
        # Reinitialize all components
        for name, ng in self.neuron_groups.items():
            # Reset membrane potential and current
            ng.v = ng.v * 0
            ng.I = ng.I * 0
            
            # Reset other state variables based on model
            if hasattr(ng, 'u'):  # For Izhikevich model
                ng.u = ng.u * 0
            if hasattr(ng, 'w'):  # For AdEx model
                ng.w = ng.w * 0
                
        # Recreate all monitors
        for name, ng in self.neuron_groups.items():
            self.spike_monitors[name] = SpikeMonitor(ng)
            
        logger.info("Reset all neural network activity")

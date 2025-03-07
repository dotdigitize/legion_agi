"""
Quantum Hamiltonian Evolution Module

This module implements Hamiltonian dynamics for quantum state evolution in the Legion AGI system.
It provides realistic physical models for quantum coherence, particularly focusing on models
relevant to quantum theories of consciousness like the Orchestrated Objective Reduction theory.
"""

import numpy as np
import scipy.linalg
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from loguru import logger

from legion_agi.config import (
    NUM_QUBITS,
    QUANTUM_SIMULATION_TIMESTEP,
    HAMILTONIAN_TYPES,
    MICROTUBULE_RESONANCE_FREQ
)


class PauliOperators:
    """Pauli operators for quantum Hamiltonian construction."""
    
    @staticmethod
    def sigma_x(n: int, i: int) -> np.ndarray:
        """
        Construct Pauli X operator for n-qubit system at position i.
        
        Args:
            n: Total number of qubits
            i: Target qubit position
            
        Returns:
            Pauli X operator for the specified qubit
        """
        if i >= n:
            raise ValueError(f"Qubit index {i} out of range for {n}-qubit system")
            
        # Single-qubit Pauli X
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        
        # Identity matrices
        id_before = np.eye(2**i, dtype=complex) if i > 0 else np.array([[1]], dtype=complex)
        id_after = np.eye(2**(n-i-1), dtype=complex) if i < n-1 else np.array([[1]], dtype=complex)
        
        # Tensor product: I_before ⊗ σ_x ⊗ I_after
        return np.kron(id_before, np.kron(sigma_x, id_after))
        
    @staticmethod
    def sigma_y(n: int, i: int) -> np.ndarray:
        """
        Construct Pauli Y operator for n-qubit system at position i.
        
        Args:
            n: Total number of qubits
            i: Target qubit position
            
        Returns:
            Pauli Y operator for the specified qubit
        """
        if i >= n:
            raise ValueError(f"Qubit index {i} out of range for {n}-qubit system")
            
        # Single-qubit Pauli Y
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        
        # Identity matrices
        id_before = np.eye(2**i, dtype=complex) if i > 0 else np.array([[1]], dtype=complex)
        id_after = np.eye(2**(n-i-1), dtype=complex) if i < n-1 else np.array([[1]], dtype=complex)
        
        # Tensor product
        return np.kron(id_before, np.kron(sigma_y, id_after))
        
    @staticmethod
    def sigma_z(n: int, i: int) -> np.ndarray:
        """
        Construct Pauli Z operator for n-qubit system at position i.
        
        Args:
            n: Total number of qubits
            i: Target qubit position
            
        Returns:
            Pauli Z operator for the specified qubit
        """
        if i >= n:
            raise ValueError(f"Qubit index {i} out of range for {n}-qubit system")
            
        # Single-qubit Pauli Z
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Identity matrices
        id_before = np.eye(2**i, dtype=complex) if i > 0 else np.array([[1]], dtype=complex)
        id_after = np.eye(2**(n-i-1), dtype=complex) if i < n-1 else np.array([[1]], dtype=complex)
        
        # Tensor product
        return np.kron(id_before, np.kron(sigma_z, id_after))


class HamiltonianGenerator:
    """Generator for quantum Hamiltonians representing physical models."""
    
    def __init__(self, num_qubits: int = NUM_QUBITS):
        """
        Initialize Hamiltonian generator.
        
        Args:
            num_qubits: Number of qubits in the system
        """
        self.num_qubits = num_qubits
        self.dimension = 2 ** num_qubits
        self.pauli = PauliOperators()
        
    def ising_model(self, 
                   J: float = 1.0, 
                   h: float = 0.5, 
                   periodic: bool = False) -> np.ndarray:
        """
        Generate Ising model Hamiltonian with transverse field.
        H = -J Σ_i σ^z_i σ^z_{i+1} - h Σ_i σ^x_i
        
        Args:
            J: Coupling strength
            h: Transverse field strength
            periodic: Use periodic boundary conditions
            
        Returns:
            Ising model Hamiltonian
        """
        n = self.num_qubits
        H = np.zeros((self.dimension, self.dimension), dtype=complex)
        
        # Add ZZ interaction terms
        for i in range(n - 1):
            H -= J * self.pauli.sigma_z(n, i) @ self.pauli.sigma_z(n, i + 1)
            
        # Add periodic boundary if requested
        if periodic and n > 2:
            H -= J * self.pauli.sigma_z(n, n - 1) @ self.pauli.sigma_z(n, 0)
            
        # Add transverse field terms
        for i in range(n):
            H -= h * self.pauli.sigma_x(n, i)
            
        return H
        
    def heisenberg_model(self, 
                        Jx: float = 1.0, 
                        Jy: float = 1.0, 
                        Jz: float = 1.0,
                        periodic: bool = False) -> np.ndarray:
        """
        Generate Heisenberg model Hamiltonian.
        H = Σ_i (Jx σ^x_i σ^x_{i+1} + Jy σ^y_i σ^y_{i+1} + Jz σ^z_i σ^z_{i+1})
        
        Args:
            Jx: X-coupling strength
            Jy: Y-coupling strength
            Jz: Z-coupling strength
            periodic: Use periodic boundary conditions
            
        Returns:
            Heisenberg model Hamiltonian
        """
        n = self.num_qubits
        H = np.zeros((self.dimension, self.dimension), dtype=complex)
        
        # Add interaction terms
        for i in range(n - 1):
            H += Jx * self.pauli.sigma_x(n, i) @ self.pauli.sigma_x(n, i + 1)
            H += Jy * self.pauli.sigma_y(n, i) @ self.pauli.sigma_y(n, i + 1)
            H += Jz * self.pauli.sigma_z(n, i) @ self.pauli.sigma_z(n, i + 1)
            
        # Add periodic boundary if requested
        if periodic and n > 2:
            H += Jx * self.pauli.sigma_x(n, n - 1) @ self.pauli.sigma_x(n, 0)
            H += Jy * self.pauli.sigma_y(n, n - 1) @ self.pauli.sigma_y(n, 0)
            H += Jz * self.pauli.sigma_z(n, n - 1) @ self.pauli.sigma_z(n, 0)
            
        return H
        
    def microtubule_model(self, 
                         omega: float = MICROTUBULE_RESONANCE_FREQ,
                         gamma: float = 0.1,
                         coupling: float = 0.3) -> np.ndarray:
        """
        Generate simplified microtubule quantum model Hamiltonian.
        Inspired by Penrose-Hameroff Orchestrated Objective Reduction theory.
        
        Args:
            omega: Oscillation frequency (resonance frequency in MHz)
            gamma: Damping coefficient
            coupling: Inter-dimer coupling strength
            
        Returns:
            Microtubule model Hamiltonian
        """
        n = self.num_qubits
        H = np.zeros((self.dimension, self.dimension), dtype=complex)
        
        # Single-dimer energy terms (simplification of tubulin conformational states)
        for i in range(n):
            H += 0.5 * omega * self.pauli.sigma_z(n, i)
            
        # Dimer-dimer coupling (simplification of dipole interactions)
        for i in range(n - 1):
            # XY-like coupling for dipole-dipole interactions
            H += coupling * self.pauli.sigma_x(n, i) @ self.pauli.sigma_x(n, i + 1)
            H += coupling * self.pauli.sigma_y(n, i) @ self.pauli.sigma_y(n, i + 1)
            
        # Circular lattice coupling (for cylindrical microtubule structure)
        if n >= 4:  # Need at least 4 qubits for reasonable cylindrical structure
            # Connect qubits in circular pattern to simulate cylindrical structure
            for i in range(n//2):
                j = i + n//2
                if j < n:
                    H += coupling * self.pauli.sigma_x(n, i) @ self.pauli.sigma_x(n, j)
                    H += coupling * self.pauli.sigma_y(n, i) @ self.pauli.sigma_y(n, j)
                    
        # Add dephasing terms (environmental interactions)
        for i in range(n):
            # Small X-field for tunneling between conformational states
            H += gamma * self.pauli.sigma_x(n, i)
            
        return H
        
    def custom_hamiltonian(self, 
                          operators: List[np.ndarray], 
                          coefficients: List[float]) -> np.ndarray:
        """
        Generate custom Hamiltonian from operators and coefficients.
        H = Σ_i coefficients[i] * operators[i]
        
        Args:
            operators: List of operator matrices
            coefficients: Corresponding coefficients
            
        Returns:
            Custom Hamiltonian
        """
        if len(operators) != len(coefficients):
            raise ValueError("Number of operators must match number of coefficients")
            
        H = np.zeros((self.dimension, self.dimension), dtype=complex)
        
        for op, coeff in zip(operators, coefficients):
            if op.shape != (self.dimension, self.dimension):
                raise ValueError(f"Operator shape {op.shape} doesn't match system dimension {self.dimension}")
                
            H += coeff * op
            
        return H
        
    def get_hamiltonian(self, hamiltonian_type: str, **params) -> np.ndarray:
        """
        Get Hamiltonian of specified type with parameters.
        
        Args:
            hamiltonian_type: Type of Hamiltonian (ising, heisenberg, microtubule)
            **params: Parameters for the Hamiltonian
            
        Returns:
            Hamiltonian matrix
        """
        if hamiltonian_type == "ising":
            J = params.get("J", HAMILTONIAN_TYPES["ising"]["parameters"]["J"])
            h = params.get("h", HAMILTONIAN_TYPES["ising"]["parameters"]["h"])
            periodic = params.get("periodic", False)
            return self.ising_model(J, h, periodic)
            
        elif hamiltonian_type == "heisenberg":
            Jx = params.get("Jx", HAMILTONIAN_TYPES["heisenberg"]["parameters"]["Jx"])
            Jy = params.get("Jy", HAMILTONIAN_TYPES["heisenberg"]["parameters"]["Jy"])
            Jz = params.get("Jz", HAMILTONIAN_TYPES["heisenberg"]["parameters"]["Jz"])
            periodic = params.get("periodic", False)
            return self.heisenberg_model(Jx, Jy, Jz, periodic)
            
        elif hamiltonian_type == "microtubule":
            omega = params.get("omega", HAMILTONIAN_TYPES["microtubule"]["parameters"]["omega"])
            gamma = params.get("gamma", HAMILTONIAN_TYPES["microtubule"]["parameters"]["gamma"])
            coupling = params.get("coupling", HAMILTONIAN_TYPES["microtubule"]["parameters"]["coupling"])
            return self.microtubule_model(omega, gamma, coupling)
            
        else:
            raise ValueError(f"Unknown Hamiltonian type: {hamiltonian_type}")


class QuantumEvolution:
    """Quantum state evolution using Hamiltonian dynamics."""
    
    def __init__(self, num_qubits: int = NUM_QUBITS):
        """
        Initialize quantum evolution simulator.
        
        Args:
            num_qubits: Number of qubits in the system
        """
        self.num_qubits = num_qubits
        self.dimension = 2 ** num_qubits
        self.hamiltonian_generator = HamiltonianGenerator(num_qubits)
        
    def calculate_time_evolution_operator(self, 
                                         hamiltonian: np.ndarray, 
                                         time: float) -> np.ndarray:
        """
        Calculate time evolution operator U(t) = exp(-i*H*t).
        
        Args:
            hamiltonian: Hamiltonian operator H
            time: Evolution time t
            
        Returns:
            Time evolution operator U(t)
        """
        # Diagonalize Hamiltonian
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        
        # Calculate exponential of diagonal matrix
        exp_diag = np.exp(-1j * eigenvalues * time)
        
        # Construct evolution operator
        U = eigenvectors @ np.diag(exp_diag) @ eigenvectors.conj().T
        
        return U
        
    def evolve_state(self, 
                    initial_state: np.ndarray, 
                    hamiltonian: np.ndarray, 
                    time: float) -> np.ndarray:
        """
        Evolve quantum state under Hamiltonian for specified time.
        
        Args:
            initial_state: Initial density matrix
            hamiltonian: Hamiltonian operator
            time: Evolution time
            
        Returns:
            Evolved density matrix
        """
        # Calculate evolution operator
        U = self.calculate_time_evolution_operator(hamiltonian, time)
        
        # Apply evolution operator
        evolved_state = U @ initial_state @ U.conj().T
        
        return evolved_state
        
    def simulate_evolution(self, 
                          initial_state: np.ndarray, 
                          hamiltonian: np.ndarray, 
                          total_time: float, 
                          time_step: float = QUANTUM_SIMULATION_TIMESTEP,
                          observables: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        """
        Simulate state evolution over time, tracking observables.
        
        Args:
            initial_state: Initial density matrix
            hamiltonian: Hamiltonian operator
            total_time: Total simulation time
            time_step: Time step size
            observables: List of observables to track
            
        Returns:
            Simulation results
        """
        # Initialize results
        times = np.arange(0, total_time, time_step)
        states = [initial_state]
        observable_values = [] if observables else None
        
        # Current state
        current_state = initial_state.copy()
        
        # Evolution operator for each time step
        U_step = self.calculate_time_evolution_operator(hamiltonian, time_step)
        
        # Initialize observable values if tracking
        if observables:
            initial_values = [np.real(np.trace(obs @ current_state)) for obs in observables]
            observable_values = [initial_values]
            
        # Simulate evolution
        for _ in range(1, len(times)):
            # Evolve state
            current_state = U_step @ current_state @ U_step.conj().T
            
            # Store state
            states.append(current_state)
            
            # Track observables
            if observables:
                values = [np.real(np.trace(obs @ current_state)) for obs in observables]
                observable_values.append(values)
                
        # Return results
        results = {
            "times": times,
            "states": states,
            "final_state": states[-1]
        }
        
        if observables:
            results["observable_values"] = np.array(observable_values)
            
        return results
        
    def evolve_state_hamiltonian(self, 
                                state: np.ndarray, 
                                hamiltonian_type: str, 
                                time_steps: int, 
                                dt: float,
                                **params) -> np.ndarray:
        """
        Evolve quantum state using specified Hamiltonian type.
        
        Args:
            state: Initial density matrix
            hamiltonian_type: Type of Hamiltonian
            time_steps: Number of time steps
            dt: Time step size
            **params: Hamiltonian parameters
            
        Returns:
            Evolved density matrix
        """
        # Get Hamiltonian
        hamiltonian = self.hamiltonian_generator.get_hamiltonian(hamiltonian_type, **params)
        
        # Simulate evolution
        total_time = time_steps * dt
        simulation = self.simulate_evolution(state, hamiltonian, total_time, dt)
        
        return simulation["final_state"]
        
    def estimate_coherence_timescale(self, 
                                   initial_state: np.ndarray,
                                   hamiltonian: np.ndarray, 
                                   decoherence_func: Callable[[np.ndarray, float], np.ndarray],
                                   coherence_measure: Callable[[np.ndarray], float],
                                   threshold: float = 0.5,
                                   max_time: float = 100.0,
                                   time_step: float = 0.1) -> float:
        """
        Estimate coherence timescale by simulating evolution with decoherence.
        
        Args:
            initial_state: Initial density matrix
            hamiltonian: Hamiltonian operator
            decoherence_func: Function that applies decoherence for a time step
            coherence_measure: Function that measures coherence of a state
            threshold: Coherence threshold as fraction of initial
            max_time: Maximum simulation time
            time_step: Time step size
            
        Returns:
            Estimated coherence time
        """
        # Calculate initial coherence
        initial_coherence = coherence_measure(initial_state)
        threshold_value = threshold * initial_coherence
        
        # Current state and time
        current_state = initial_state.copy()
        current_time = 0.0
        
        # Track coherence values
        times = [0.0]
        coherence_values = [initial_coherence]
        
        # Evolution operator for each time step
        U_step = self.calculate_time_evolution_operator(hamiltonian, time_step)
        
        while current_time < max_time:
            # Evolve state with Hamiltonian
            current_state = U_step @ current_state @ U_step.conj().T
            
            # Apply decoherence
            current_state = decoherence_func(current_state, time_step)
            
            # Update time
            current_time += time_step
            times.append(current_time)
            
            # Measure coherence
            coherence = coherence_measure(current_state)
            coherence_values.append(coherence)
            
            # Check threshold
            if coherence < threshold_value:
                return current_time
                
        # Coherence remained above threshold
        return max_time
        
    def simulate_microtubule_oscillations(self, 
                                        initial_state: np.ndarray,
                                        duration_ms: float,
                                        temperature_kelvin: float = 300.0) -> Dict[str, Any]:
        """
        Simulate quantum oscillations in microtubules with realistic parameters.
        Based on theories of quantum consciousness like Orchestrated Objective Reduction.
        
        Args:
            initial_state: Initial density matrix
            duration_ms: Simulation duration in milliseconds
            temperature_kelvin: Temperature in Kelvin
            
        Returns:
            Simulation results
        """
        # Create microtubule Hamiltonian
        # Parameters derived from theoretical models of microtubule dynamics
        omega = MICROTUBULE_RESONANCE_FREQ  # ~8 MHz resonance frequency
        
        # Temperature-dependent damping
        # Higher temperature = stronger coupling to environment = faster decoherence
        gamma = 0.1 * (temperature_kelvin / 310.0)**2  # Scale with temperature^2
        
        # Inter-dimer coupling
        coupling = 0.3
        
        hamiltonian = self.hamiltonian_generator.microtubule_model(omega, gamma, coupling)
        
        # Define observables to track
        # 1. Energy - Hamiltonian expectation
        # 2. Coherence between first two tubulin dimers - σ_x1 σ_x2 + σ_y1 σ_y2
        observables = [
            hamiltonian,  # Energy
            self.hamiltonian_generator.pauli.sigma_x(self.num_qubits, 0) @ 
                self.hamiltonian_generator.pauli.sigma_x(self.num_qubits, 1) +
            self.hamiltonian_generator.pauli.sigma_y(self.num_qubits, 0) @ 
                self.hamiltonian_generator.pauli.sigma_y(self.num_qubits, 1)  # Coherence
        ]
        
        # Temperature-dependent decoherence time
        # Based on theoretical estimates for warm wet brain (~picoseconds at body temp)
        # We scale it down with temperature
        decoherence_time = 1.0 * (310.0 / temperature_kelvin)**2  # ms
        
        # Decoherence function
        def decoherence(state: np.ndarray, dt: float) -> np.ndarray:
            # Simple exponential decoherence model
            # Diagonal elements unchanged, off-diagonal decays
            result = state.copy()
            for i in range(self.dimension):
                for j in range(self.dimension):
                    if i != j:
                        decay = np.exp(-dt / decoherence_time)
                        result[i, j] *= decay
            return result
            
        # Time step for simulation (smaller of 1/10 of decoherence time or 0.1 ms)
        dt = min(decoherence_time / 10, 0.1)
        
        # Run simulation
        simulation = self.simulate_evolution(
            initial_state,
            hamiltonian,
            duration_ms,
            dt,
            observables
        )
        
        # Extract results
        times = simulation["times"]
        states = simulation["states"]
        energy_values = simulation["observable_values"][:, 0] if "observable_values" in simulation else None
        coherence_values = simulation["observable_values"][:, 1] if "observable_values" in simulation else None
        
        # Calculate coherence decay rate
        if coherence_values is not None and len(coherence_values) > 1:
            initial_coherence = coherence_values[0]
            final_coherence = coherence_values[-1]
            coherence_decay_rate = (initial_coherence - final_coherence) / duration_ms
        else:
            coherence_decay_rate = None
            
        # Results
        results = {
            "times_ms": times,
            "states": states,
            "final_state": states[-1],
            "energy_values": energy_values,
            "coherence_values": coherence_values,
            "coherence_decay_rate": coherence_decay_rate,
            "temperature_kelvin": temperature_kelvin,
            "decoherence_time_ms": decoherence_time,
            "resonance_frequency_mhz": omega
        }
        
        logger.info(
            f"Simulated microtubule quantum oscillations: {duration_ms}ms at {temperature_kelvin}K, "
            f"decoherence time: {decoherence_time:.3f}ms"
        )
        
        return results

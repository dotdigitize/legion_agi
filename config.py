"""
Configuration file for Legion AGI system.
Contains all configurable parameters for the system.
"""

import os
from pathlib import Path

# System paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
DB_DIR = BASE_DIR / "db"

# Ensure directories exist
for directory in [DATA_DIR, LOG_DIR, DB_DIR]:
    os.makedirs(directory, exist_ok=True)

# LLM Configuration
DEFAULT_MODEL = "llama3.1:8b"
INFERENCE_TEMPERATURE = 0.7
INFERENCE_TOP_P = 0.9
MAX_TOKENS = 2000

# Agent Configuration
MAX_AGENTS = 10
AGENT_MEMORY_CAPACITY = 50  # Number of items in short-term memory
CONSOLIDATION_THRESHOLD = 5  # Number of items before memory consolidation
AGENT_REASONING_STEPS = 10   # Maximum reasoning steps per agent

# Quantum Memory Configuration
NUM_QUBITS = 4  # Default number of qubits for quantum memory
DECOHERENCE_RATE = 0.01  # Rate at which quantum states decohere
ENTANGLEMENT_STRENGTH = 0.8  # Strength of entanglement between memory units

# Enhanced Quantum Parameters
TEMPERATURE_KELVIN = 300.0  # Environmental temperature (room temperature)
COHERENCE_TIME_MS = 100.0  # Characteristic coherence time in milliseconds
QUANTUM_SIMULATION_TIMESTEP = 0.1  # Default timestep for quantum simulation (ms)
MICROTUBULE_RESONANCE_FREQ = 8.085  # Resonance frequency (MHz) for microtubules
COHERENCE_THRESHOLD = 0.3  # Threshold for quantum vs classical processing

# Spiking Neural Network Configuration
SNN_SIMULATION_TIME = 100  # milliseconds
SNN_INTEGRATION_TIME = 10   # milliseconds
SNN_THRESHOLD = 1.0        # Firing threshold
SNN_RESET = 0.0           # Reset potential
SNN_REFRACTORY_PERIOD = 5  # milliseconds

# Global Workspace Configuration
GW_COMPETITION_THRESHOLD = 0.7  # Threshold for information to enter global workspace
GW_BROADCAST_CYCLES = 3        # Number of cycles for global broadcast
GW_WORKSPACE_CAPACITY = 7      # Capacity based on Miller's Law (7±2)
GW_CONSCIOUSNESS_THRESHOLD = 0.6  # Threshold for "conscious access"

# Database Configuration
DB_FILENAME = "legion_agi.db"
DB_PATH = DB_DIR / DB_FILENAME

# Logging Configuration
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = LOG_DIR / "legion_agi.log"
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
LOG_ROTATION = "10 MB"

# Evolution Parameters
MUTATION_RATE = 0.05     # Rate of random mutations during evolution
CROSSOVER_RATE = 0.3     # Rate of trait crossovers during evolution
SELECTION_PRESSURE = 0.7  # Strength of selection pressure
GENERATIONS_PER_CYCLE = 5  # Number of generations per evolution cycle

# Mode Configuration
TOOL_MODE = "tool"       # Single-use mode
EVOLUTION_MODE = "evolve"  # Continuous evolution mode

# Reasoning Methods Parameters
PAST_DEPTH = 3           # Depth of persona/action/solution/task analysis
RAFT_ITERATIONS = 5      # Number of reasoning/analysis/feedback/thought iterations
EAT_EVALUATION_THRESHOLD = 0.7  # Threshold for solution acceptance

# Quantum Hamiltonian Parameters
# These parameters define default Hamiltonians for quantum evolution
HAMILTONIAN_TYPES = {
    "ising": {
        "description": "Ising model with transverse field",
        "parameters": {
            "J": 1.0,  # Coupling strength
            "h": 0.5   # Transverse field strength
        }
    },
    "heisenberg": {
        "description": "Heisenberg model for spin interactions",
        "parameters": {
            "Jx": 1.0,  # X coupling
            "Jy": 1.0,  # Y coupling
            "Jz": 1.0   # Z coupling
        }
    },
    "microtubule": {
        "description": "Simplified microtubule quantum oscillations",
        "parameters": {
            "omega": MICROTUBULE_RESONANCE_FREQ,  # Resonance frequency (MHz)
            "gamma": 0.1,  # Damping coefficient
            "coupling": 0.3  # Inter-dimer coupling
        }
    }
}

# Von Neumann Algebra Configuration
VON_NEUMANN_CONFIG = {
    "projection_lattice_depth": 3,  # Depth of projection lattice
    "quantum_logic_gates": ["AND", "OR", "NOT", "XOR"],  # Supported logical operations
    "operator_precision": 1e-10,  # Numerical precision for operator calculations
    "eigenvalue_threshold": 1e-8  # Threshold for treating eigenvalues as zero
}

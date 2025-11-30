# Path: QAgents-workflos/tests/test_problems.py
# Relations: Used by evaluation_harness.py, run_evaluation.py
# Description: Real quantum computing problems requiring LLM reasoning
#              Each problem has increasing complexity and real-world relevance
"""
Test Problems Module: Real Quantum Computing Challenges

TESTING FRAMEWORK DESIGN:
=========================

Each problem requires actual LLM reasoning to solve - no hardcoded templates.
The LLM must understand the quantum mechanics and generate appropriate QASM.

EVALUATION MODES:
-----------------
1. NAKED: 1 LLM call per problem (direct reasoning, no agents)
2. GUIDED: 1 + 4 LLM calls (initial + architect/builder/validator/scorer agents)  
3. BLACKBOARD: 1 + 8-12 LLM calls (initial + collaborative agent rounds)

PROBLEM CATEGORIES:
-------------------
EASY (1-2 qubits, 1-3 gates):
  - Fundamental single/two-qubit operations
  - Direct QASM generation possible

MEDIUM (2-3 qubits, 4-8 gates):
  - Require understanding of gate decomposition
  - Multiple valid solutions possible

HARD (3+ qubits, 8+ gates):
  - Algorithm implementation
  - Optimization considerations
  - Real-world applications
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class ProblemDifficulty(Enum):
    """Problem difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"  # New: Push NAKED to its limits


class ProblemCategory(Enum):
    """Problem categories for research tracking."""
    STATE_PREPARATION = "state_prep"
    GATE_SYNTHESIS = "gate_synthesis"
    ALGORITHM = "algorithm"
    ERROR_CORRECTION = "error_correction"
    OPTIMIZATION = "optimization"


@dataclass
class ExpectedOutput:
    """Expected output for validation."""
    min_qubits: int
    max_qubits: int = 10
    max_depth: Optional[int] = None
    required_gates: List[str] = field(default_factory=list)
    forbidden_gates: List[str] = field(default_factory=list)
    expected_states: Dict[str, float] = field(default_factory=dict)
    tolerance: float = 0.1  # Probability tolerance for state matching
    must_be_unitary: bool = True
    hardware_compatible: bool = True


@dataclass
class TestProblem:
    """A quantum circuit test problem for LLM evaluation."""
    id: str
    name: str
    description: str

    # The prompt sent to the LLM - must require reasoning
    prompt: str

    # Category and difficulty for analysis
    difficulty: ProblemDifficulty
    category: ProblemCategory

    # Validation criteria
    expected: ExpectedOutput

    # Metadata for research tracking
    tags: List[str] = field(default_factory=list)
    reference_solution: Optional[str] = None  # Known optimal QASM
    optimal_depth: Optional[int] = None
    optimal_gate_count: Optional[int] = None

    # Research tracking
    requires_understanding: List[str] = field(default_factory=list)
    common_mistakes: List[str] = field(default_factory=list)

    @property
    def goal(self) -> str:
        """Alias for prompt - used by orchestrators."""
        return self.prompt
# =============================================================================
# EASY PROBLEMS: Fundamental Quantum Operations
# =============================================================================

PROBLEM_E1_PHASE_FLIP = TestProblem(
    id="easy_001",
    name="Phase Flip State",
    description="Create the |−⟩ state (phase-flipped superposition)",
    prompt="""Create a quantum circuit that prepares the |−⟩ state.

The |−⟩ state is defined as: (|0⟩ - |1⟩)/√2

This is different from the |+⟩ state which is (|0⟩ + |1⟩)/√2.

Requirements:
- Use a single qubit
- The final state should have equal probability of 0 and 1
- But the relative phase between them should be π (negative)

Provide the OpenQASM 2.0 circuit.""",
    difficulty=ProblemDifficulty.EASY,
    category=ProblemCategory.STATE_PREPARATION,
    expected=ExpectedOutput(
        min_qubits=1,
        max_qubits=1,
        max_depth=2,
        required_gates=["h", "z"],  # or x then h
        expected_states={"0": 0.5, "1": 0.5}
    ),
    tags=["superposition", "phase", "single-qubit"],
    requires_understanding=["Hadamard gate", "Z gate", "quantum phases"],
    common_mistakes=["Using only H (creates |+⟩ not |−⟩)", "Wrong gate order"],
    optimal_depth=2,
    optimal_gate_count=2
)

PROBLEM_E2_CONTROLLED_NOT = TestProblem(
    id="easy_002",
    name="Entanglement Generation",
    description="Create maximal entanglement between two qubits",
    prompt="""Create a quantum circuit that maximally entangles two qubits.

Starting from |00⟩, create the Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2

Requirements:
- Use exactly 2 qubits
- Measuring both qubits should give 00 or 11 with equal probability
- The qubits must be entangled (not just in superposition)

Think about what gates create entanglement.
Provide the OpenQASM 2.0 circuit.""",
    difficulty=ProblemDifficulty.EASY,
    category=ProblemCategory.STATE_PREPARATION,
    expected=ExpectedOutput(
        min_qubits=2,
        max_qubits=2,
        max_depth=3,
        required_gates=["h", "cx"],
        expected_states={"00": 0.5, "11": 0.5}
    ),
    tags=["entanglement", "bell", "cnot"],
    requires_understanding=["Hadamard gate", "CNOT gate", "entanglement"],
    common_mistakes=["Applying H to both qubits (no entanglement)", "Wrong CNOT direction"],
    optimal_depth=2,
    optimal_gate_count=2
)

PROBLEM_E3_MEASUREMENT_BASIS = TestProblem(
    id="easy_003", 
    name="X-Basis Measurement Prep",
    description="Prepare a state for X-basis measurement",
    prompt="""Create a circuit that transforms a Z-basis state into X-basis.

Starting with |0⟩, prepare the state so that if we were to measure in the 
X-basis (instead of Z-basis), we would get |+⟩ deterministically.

In other words: Transform |0⟩ → |+⟩ where |+⟩ = (|0⟩ + |1⟩)/√2

Requirements:
- Single qubit circuit
- The state should be the +1 eigenstate of the X operator

Provide the OpenQASM 2.0 circuit.""",
    difficulty=ProblemDifficulty.EASY,
    category=ProblemCategory.STATE_PREPARATION,
    expected=ExpectedOutput(
        min_qubits=1,
        max_qubits=1,
        max_depth=1,
        required_gates=["h"],
        expected_states={"0": 0.5, "1": 0.5}
    ),
    tags=["basis-change", "hadamard", "measurement"],
    requires_understanding=["Measurement bases", "Hadamard as basis change"],
    common_mistakes=["Not understanding basis transformation"],
    optimal_depth=1,
    optimal_gate_count=1
)


# =============================================================================
# MEDIUM PROBLEMS: Gate Decomposition and Multi-Qubit Operations
# =============================================================================

PROBLEM_M1_SWAP_DECOMPOSITION = TestProblem(
    id="medium_001",
    name="SWAP from CNOTs",
    description="Implement SWAP gate using only CNOT gates",
    prompt="""Decompose the SWAP gate into basic gates.

The SWAP gate exchanges the states of two qubits:
SWAP|ab⟩ = |ba⟩

You must implement SWAP using only CNOT gates (no native SWAP allowed).

Requirements:
- Use exactly 2 qubits
- Only use CNOT (cx) gates - no other two-qubit gates
- The circuit should swap the state of qubit 0 and qubit 1
- Test: if input is |01⟩, output should be |10⟩

Hint: CNOT can be thought of as conditional bit flip.

Provide the OpenQASM 2.0 circuit.""",
    difficulty=ProblemDifficulty.MEDIUM,
    category=ProblemCategory.GATE_SYNTHESIS,
    expected=ExpectedOutput(
        min_qubits=2,
        max_qubits=2,
        max_depth=6,
        required_gates=["cx"],
        forbidden_gates=["swap"]
    ),
    tags=["decomposition", "swap", "cnot-only"],
    requires_understanding=["CNOT behavior", "Gate decomposition"],
    common_mistakes=["Wrong number of CNOTs", "Wrong CNOT directions"],
    reference_solution="OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncx q[0],q[1];\ncx q[1],q[0];\ncx q[0],q[1];",
    optimal_depth=3,
    optimal_gate_count=3
)

PROBLEM_M2_CONTROLLED_Z = TestProblem(
    id="medium_002",
    name="CZ from Basic Gates",
    description="Build Controlled-Z using H and CNOT",
    prompt="""Implement the Controlled-Z (CZ) gate using only Hadamard and CNOT gates.

The CZ gate applies a Z gate to the target qubit when the control is |1⟩:
CZ|00⟩ = |00⟩
CZ|01⟩ = |01⟩  
CZ|10⟩ = |10⟩
CZ|11⟩ = -|11⟩  (note the phase flip!)

Requirements:
- Use only H and CNOT gates
- No native CZ gate allowed
- 2 qubits

Hint: Think about how H transforms Z operations.

Provide the OpenQASM 2.0 circuit.""",
    difficulty=ProblemDifficulty.MEDIUM,
    category=ProblemCategory.GATE_SYNTHESIS,
    expected=ExpectedOutput(
        min_qubits=2,
        max_qubits=2,
        max_depth=5,
        required_gates=["h", "cx"],
        forbidden_gates=["cz"]
    ),
    tags=["decomposition", "controlled-z", "phase"],
    requires_understanding=["CZ gate definition", "H-Z-H = X identity"],
    common_mistakes=["Forgetting H gates", "Wrong qubit as target"],
    reference_solution="OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nh q[1];\ncx q[0],q[1];\nh q[1];",
    optimal_depth=3,
    optimal_gate_count=3
)

PROBLEM_M3_PHASE_ESTIMATION_PREP = TestProblem(
    id="medium_003",
    name="Phase Kickback Setup",
    description="Create the phase kickback configuration",
    prompt="""Create a circuit demonstrating quantum phase kickback.

Phase kickback is a key concept where applying a controlled-U gate
causes the control qubit to acquire the eigenvalue phase.

Setup:
1. Prepare control qubit in |+⟩ superposition
2. Prepare target qubit in |1⟩ (eigenstate of Z with eigenvalue -1)
3. Apply CZ gate
4. The control qubit should now be in |−⟩ state

The final state of the control qubit (q[0]) should show the phase kickback.

Requirements:
- 2 qubits
- Control in superposition, target in |1⟩
- Apply controlled operation
- Use only basic gates (H, X, CX, CZ allowed)

Provide the OpenQASM 2.0 circuit.""",
    difficulty=ProblemDifficulty.MEDIUM,
    category=ProblemCategory.ALGORITHM,
    expected=ExpectedOutput(
        min_qubits=2,
        max_qubits=2,
        max_depth=5,
        required_gates=["h", "x"],
        expected_states={"01": 0.5, "11": 0.5}  # After kickback
    ),
    tags=["phase-kickback", "algorithm-primitive", "phase-estimation"],
    requires_understanding=["Phase kickback", "Eigenstates", "Controlled operations"],
    common_mistakes=["Target not in eigenstate", "Missing superposition"],
    optimal_depth=4,
    optimal_gate_count=4
)


# =============================================================================
# HARD PROBLEMS: Algorithm Implementation
# =============================================================================

PROBLEM_H1_DEUTSCH = TestProblem(
    id="hard_001",
    name="Deutsch Algorithm",
    description="Implement Deutsch's algorithm for function type detection",
    prompt="""Implement Deutsch's algorithm to determine if a function is constant or balanced.

Deutsch's algorithm determines whether a black-box function f:{0,1}→{0,1} is:
- Constant: f(0)=f(1) (always 0 or always 1)
- Balanced: f(0)≠f(1) (different outputs)

For this problem, implement the oracle for the BALANCED function f(x) = x.

Algorithm structure:
1. Initialize |01⟩ (input qubit |0⟩, ancilla qubit |1⟩)
2. Apply H to both qubits
3. Apply the oracle Uf: |x,y⟩ → |x, y⊕f(x)⟩
4. Apply H to the input qubit
5. Measure input qubit: |1⟩ means balanced

For f(x)=x, the oracle is just a CNOT.

Requirements:
- 2 qubits
- Implement full Deutsch circuit with f(x)=x oracle
- After measurement, input qubit should be in |1⟩

Provide the OpenQASM 2.0 circuit.""",
    difficulty=ProblemDifficulty.HARD,
    category=ProblemCategory.ALGORITHM,
    expected=ExpectedOutput(
        min_qubits=2,
        max_qubits=2,
        max_depth=8,
        required_gates=["h", "x", "cx"],
        expected_states={"11": 1.0}  # Input qubit is 1 (balanced), ancilla is 1
    ),
    tags=["algorithm", "deutsch", "oracle"],
    requires_understanding=["Deutsch algorithm", "Oracle construction", "Interference"],
    common_mistakes=["Wrong initial state", "Missing ancilla preparation", "Oracle errors"],
    optimal_depth=5,
    optimal_gate_count=6
)

PROBLEM_H2_GROVER_2QUBIT = TestProblem(
    id="hard_002",
    name="Grover Search (2-qubit)",
    description="Find marked state |11⟩ using Grover's algorithm",
    prompt="""Implement 2-qubit Grover's search algorithm to find the state |11⟩.

Grover's algorithm amplifies the probability of the marked state.

For 2 qubits with 1 marked state, we need exactly 1 iteration:

1. Initialize: H⊗H on |00⟩ → equal superposition
2. Oracle: Mark |11⟩ with a phase flip (multiply by -1)
3. Diffusion: Reflect about the average amplitude

Oracle for |11⟩: Apply CZ (or equivalent)
Diffusion operator: H⊗H · (2|00⟩⟨00| - I) · H⊗H

Requirements:
- 2 qubits
- After 1 Grover iteration, |11⟩ should have probability ≈ 1
- Use only basic gates

Provide the OpenQASM 2.0 circuit.""",
    difficulty=ProblemDifficulty.HARD,
    category=ProblemCategory.ALGORITHM,
    expected=ExpectedOutput(
        min_qubits=2,
        max_qubits=2,
        max_depth=12,
        required_gates=["h", "x", "cx"],
        expected_states={"11": 1.0},
        tolerance=0.1
    ),
    tags=["algorithm", "grover", "search", "amplitude-amplification"],
    requires_understanding=["Grover's algorithm", "Oracle design", "Diffusion operator"],
    common_mistakes=["Wrong oracle phase", "Missing diffusion", "Too many/few iterations"],
    optimal_depth=8,
    optimal_gate_count=10
)

PROBLEM_H3_TELEPORTATION_PREP = TestProblem(
    id="hard_003",
    name="Quantum Teleportation Setup",
    description="Prepare the entangled resource state for teleportation",
    prompt="""Create the initial setup for quantum teleportation.

Quantum teleportation requires:
1. The state to teleport |ψ⟩ on qubit 0
2. A shared Bell pair between qubits 1 and 2

For this problem:
- Prepare qubit 0 in state |+⟩ (the state we'll "teleport")
- Prepare qubits 1 and 2 in the Bell state (|00⟩ + |11⟩)/√2
- Qubit 1 goes to Alice (sender), qubit 2 to Bob (receiver)

Requirements:
- 3 qubits
- q[0]: |+⟩ state (to be teleported)
- q[1], q[2]: Bell pair (shared entanglement)

After this setup, Alice has q[0] and q[1], Bob has q[2].

Provide the OpenQASM 2.0 circuit.""",
    difficulty=ProblemDifficulty.HARD,
    category=ProblemCategory.ALGORITHM,
    expected=ExpectedOutput(
        min_qubits=3,
        max_qubits=3,
        max_depth=4,
        required_gates=["h", "cx"]
    ),
    tags=["algorithm", "teleportation", "entanglement", "bell-state"],
    requires_understanding=["Quantum teleportation", "Bell states", "Entanglement as resource"],
    common_mistakes=["Wrong qubits entangled", "State to teleport not prepared"],
    optimal_depth=3,
    optimal_gate_count=4
)


# =============================================================================
# PROBLEM SETS
# =============================================================================

EASY_PROBLEMS = [
    PROBLEM_E1_PHASE_FLIP,
    PROBLEM_E2_CONTROLLED_NOT,
    PROBLEM_E3_MEASUREMENT_BASIS
]

MEDIUM_PROBLEMS = [
    PROBLEM_M1_SWAP_DECOMPOSITION,
    PROBLEM_M2_CONTROLLED_Z,
    PROBLEM_M3_PHASE_ESTIMATION_PREP
]

HARD_PROBLEMS = [
    PROBLEM_H1_DEUTSCH,
    PROBLEM_H2_GROVER_2QUBIT,
    PROBLEM_H3_TELEPORTATION_PREP
]


# ============================================================================
# VERY_HARD PROBLEMS: Push NAKED to its limits
# ============================================================================

PROBLEM_VH1_QFT_4QUBIT = TestProblem(
    id="very_hard_001",
    name="4-Qubit QFT",
    description="Implement full Quantum Fourier Transform on 4 qubits",
    prompt="""Implement the complete Quantum Fourier Transform (QFT) on 4 qubits.

The QFT transforms computational basis states into Fourier basis:
QFT|x⟩ = (1/√N) Σ_{k=0}^{N-1} e^{2πixk/N} |k⟩

For 4 qubits (N=16), the circuit requires:
1. Apply Hadamard to each qubit in sequence
2. Apply controlled phase rotations (CR_k) between qubits
3. SWAP qubits to correct bit ordering (optional for some conventions)

Phase rotation angles: R_k = rotation by π/2^(k-1)
- R_2 = π/2 (S gate or cp(π/2))
- R_3 = π/4 (T gate or cp(π/4))
- R_4 = π/8 (cp(π/8))

Requirements:
- Use exactly 4 qubits
- Must use H, controlled-phase (cp or crz), and optionally SWAP gates
- Do NOT use QFT as a black box - implement the full decomposition
- Include proper phase rotations between all qubit pairs

The output should show interference patterns in the Fourier basis.

Provide the OpenQASM 2.0 circuit.""",
    difficulty=ProblemDifficulty.VERY_HARD,
    category=ProblemCategory.ALGORITHM,
    expected=ExpectedOutput(
        min_qubits=4,
        max_qubits=4,
        max_depth=20,
        required_gates=["h"]
    ),
    tags=["qft", "fourier", "phase-rotation", "multi-qubit"],
    requires_understanding=["QFT algorithm", "Controlled phase gates", "Bit reversal"],
    common_mistakes=["Wrong phase angles", "Missing controlled rotations", "Forgetting bit reversal"],
    optimal_depth=12,
    optimal_gate_count=16
)

PROBLEM_VH2_GROVER_3QUBIT = TestProblem(
    id="very_hard_002",
    name="Grover 3-Qubit Search",
    description="Implement Grover's search on 3 qubits with 2 iterations",
    prompt="""Implement 3-qubit Grover's search algorithm to find the marked state |101⟩.

For 3 qubits (N=8 states), the optimal number of iterations is approximately π√N/4 ≈ 2.

Algorithm structure (repeat 2 times):
1. Initial superposition: H⊗H⊗H on |000⟩

For EACH Grover iteration:
2. Oracle: Mark |101⟩ with phase flip (multiply amplitude by -1)
   - Oracle for |101⟩: X on q[1], then CCZ (or Toffoli+phase), then X on q[1]
   - Alternative: use multi-controlled Z gate
   
3. Diffusion operator (Grover diffuser):
   - Apply H to all qubits
   - Apply X to all qubits  
   - Apply multi-controlled Z (CCZ or decomposition)
   - Apply X to all qubits
   - Apply H to all qubits

Requirements:
- Use exactly 3 qubits
- Implement BOTH oracle and diffusion operator
- Perform exactly 2 Grover iterations
- After 2 iterations, |101⟩ should have probability > 0.9
- Use basic gates: H, X, CX, CCX (Toffoli), CZ, or their equivalents

IMPORTANT: You must implement CCZ using either:
- ccx followed by cz and ccx (Toffoli-based)
- h on target, ccx, h on target (standard decomposition)

Provide the OpenQASM 2.0 circuit.""",
    difficulty=ProblemDifficulty.VERY_HARD,
    category=ProblemCategory.ALGORITHM,
    expected=ExpectedOutput(
        min_qubits=3,
        max_qubits=3,
        max_depth=30,
        required_gates=["h", "x", "cx"],
        expected_states={"101": 0.9},
        tolerance=0.15
    ),
    tags=["grover", "search", "oracle", "diffusion", "multi-iteration"],
    requires_understanding=["Grover's algorithm", "Multi-controlled gates", "Oracle design", "Diffusion operator"],
    common_mistakes=["Wrong oracle", "Single iteration only", "Incorrect diffusion", "Missing CCZ decomposition"],
    optimal_depth=24,
    optimal_gate_count=40
)

PROBLEM_VH3_VQE_ANSATZ = TestProblem(
    id="very_hard_003",
    name="VQE Hardware-Efficient Ansatz",
    description="Construct a 4-qubit hardware-efficient ansatz for VQE",
    prompt="""Construct a 4-qubit hardware-efficient variational ansatz for VQE.

A hardware-efficient ansatz is a parameterized quantum circuit used in VQE
(Variational Quantum Eigensolver) to prepare trial wavefunctions.

Structure (2 layers):

LAYER 1:
1. Apply Ry(θ) rotations to all 4 qubits (use ry gate with parameter, e.g., ry(pi/4))
2. Apply Rz(φ) rotations to all 4 qubits (use rz gate with parameter, e.g., rz(pi/4))
3. Apply entangling CNOT ladder: cx q[0],q[1]; cx q[1],q[2]; cx q[2],q[3];

LAYER 2:
4. Apply Ry(θ') rotations to all 4 qubits
5. Apply Rz(φ') rotations to all 4 qubits
6. Apply entangling CNOT ladder again

For this implementation, use fixed angles:
- Layer 1: ry(0.5) and rz(0.3) on all qubits
- Layer 2: ry(0.7) and rz(0.2) on all qubits

Requirements:
- Use exactly 4 qubits
- Implement 2 full layers (rotation + entanglement each)
- Use ry, rz, and cx gates
- Linear entanglement pattern (nearest-neighbor CNOTs)

This circuit structure is used on real quantum hardware (IBM, Google) for
quantum chemistry and optimization problems.

Provide the OpenQASM 2.0 circuit.""",
    difficulty=ProblemDifficulty.VERY_HARD,
    category=ProblemCategory.ALGORITHM,
    expected=ExpectedOutput(
        min_qubits=4,
        max_qubits=4,
        max_depth=16,
        required_gates=["ry", "rz", "cx"]
    ),
    tags=["vqe", "ansatz", "variational", "quantum-chemistry", "hardware-efficient"],
    requires_understanding=["VQE algorithm", "Parameterized circuits", "Hardware constraints", "Entanglement layers"],
    common_mistakes=["Missing rotation layers", "Wrong entanglement pattern", "Incorrect parameter format"],
    optimal_depth=12,
    optimal_gate_count=22
)

PROBLEM_VH4_BERNSTEIN_VAZIRANI = TestProblem(
    id="very_hard_004",
    name="Bernstein-Vazirani 4-bit",
    description="Implement Bernstein-Vazirani algorithm to find hidden string s=1011",
    prompt="""Implement the Bernstein-Vazirani algorithm to find the hidden string s=1011.

The Bernstein-Vazirani algorithm finds a hidden n-bit string s in ONE query.
Given a function f(x) = s·x mod 2 (bitwise dot product), find s.

For s=1011 (4 bits), we need 5 qubits (4 input + 1 ancilla):

Algorithm:
1. Initialize all input qubits to |0⟩, ancilla to |1⟩
2. Apply H to all 5 qubits (creates superposition + phase kickback setup)
3. Apply Oracle U_f: For each bit s_i=1, apply CNOT from q[i] to ancilla
   - s=1011 means: CNOT from q[0] to q[4], q[2] to q[4], q[3] to q[4]
   - (s[0]=1, s[1]=0, s[2]=1, s[3]=1 → control qubits 0, 2, 3)
4. Apply H to all input qubits (NOT the ancilla)
5. Measure input qubits → reveals s directly

Requirements:
- Use 5 qubits (q[0-3] for input, q[4] for ancilla)
- Prepare ancilla in |1⟩ state before Hadamards
- Oracle: CNOT from q[0], q[2], q[3] to q[4] (positions where s has 1)
- Apply final Hadamards only to input qubits
- Measure input qubits → should give |1011⟩

After measurement, the input register should read 1011 with probability 1.0.

Provide the OpenQASM 2.0 circuit.""",
    difficulty=ProblemDifficulty.VERY_HARD,
    category=ProblemCategory.ALGORITHM,
    expected=ExpectedOutput(
        min_qubits=5,
        max_qubits=5,
        max_depth=10,
        required_gates=["h", "x", "cx"],
        expected_states={"10111": 1.0},  # 1011 in input register, 1 in ancilla
        tolerance=0.05
    ),
    tags=["bernstein-vazirani", "oracle", "hidden-string", "query-complexity"],
    requires_understanding=["Bernstein-Vazirani algorithm", "Oracle construction", "Phase kickback"],
    common_mistakes=["Wrong oracle CNOTs", "Missing ancilla preparation", "Hadamards on ancilla"],
    optimal_depth=6,
    optimal_gate_count=15
)

VERY_HARD_PROBLEMS = [
    PROBLEM_VH1_QFT_4QUBIT,
    PROBLEM_VH2_GROVER_3QUBIT,
    PROBLEM_VH3_VQE_ANSATZ,
    PROBLEM_VH4_BERNSTEIN_VAZIRANI
]

ALL_PROBLEMS = EASY_PROBLEMS + MEDIUM_PROBLEMS + HARD_PROBLEMS + VERY_HARD_PROBLEMS

# Problem registry by ID
PROBLEMS_BY_ID = {p.id: p for p in ALL_PROBLEMS}


def get_problem(problem_id: str) -> Optional[TestProblem]:
    """Get a problem by ID."""
    return PROBLEMS_BY_ID.get(problem_id)


def get_problems_by_difficulty(difficulty: ProblemDifficulty) -> List[TestProblem]:
    """Get all problems of a specific difficulty."""
    # Handle string input
    if isinstance(difficulty, str):
        difficulty = ProblemDifficulty(difficulty.lower())
    return [p for p in ALL_PROBLEMS if p.difficulty == difficulty]


def get_problems_by_category(category: ProblemCategory) -> List[TestProblem]:
    """Get all problems of a specific category."""
    return [p for p in ALL_PROBLEMS if p.category == category]


def get_problems_by_tag(tag: str) -> List[TestProblem]:
    """Get all problems with a specific tag."""
    return [p for p in ALL_PROBLEMS if tag in p.tags]


def get_research_problem_set() -> List[TestProblem]:
    """Get the standard research evaluation set (3 problems, one per difficulty)."""
    return [
        PROBLEM_E1_PHASE_FLIP,      # Easy: Phase flip state
        PROBLEM_M1_SWAP_DECOMPOSITION,  # Medium: SWAP decomposition
        PROBLEM_H1_DEUTSCH          # Hard: Deutsch algorithm
    ]

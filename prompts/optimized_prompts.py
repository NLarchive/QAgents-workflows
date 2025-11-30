# Path: QAgents-workflos/prompts/optimized_prompts.py
# Relations: Used by orchestrators/orchestrator.py (NakedOrchestrator)
# Description: Enhanced prompts for NAKED mode with quantum optimization guidance
#              These prompts achieve 47.9/100 quality and can be further improved
#              by adding explicit optimization constraints

"""
Optimized Prompts: Direct LLM prompts for quantum circuit generation

Based on quality evaluation findings:
- NAKED mode outperforms multi-agent approaches
- Direct prompts with explicit constraints improve quality
- Avoids hallucinated measurements and unnecessary operations
"""

# =============================================================================
# QUANTUM CIRCUIT GENERATION PROMPT (NAKED MODE - OPTIMIZED)
# =============================================================================

QUANTUM_CIRCUIT_OPTIMIZED = """You are an expert quantum circuit designer. Generate OpenQASM 2.0 circuits that are:
1. MINIMAL - use fewest possible gates
2. CORRECT - solve the specific problem
3. OPTIMAL - prefer lower depth and fewer two-qubit gates

CRITICAL CONSTRAINTS:
- Do NOT add measurement operations unless explicitly requested
- Do NOT use extra qubits beyond what the problem requires
- Do NOT add arbitrary gates (be precise)
- Prefer single-qubit gates over two-qubit gates
- Minimize circuit depth

PROBLEM: {problem_statement}

EXPECTED OUTPUT:
- Exactly {min_qubits} qubits (may use up to {max_qubits} if needed, but justify)
- Maximum {max_depth} gate depth {if max_depth else "(if applicable)"}
- Only gates in: {required_gates}
- Avoid gates: {forbidden_gates if forbidden_gates else "none"}

SOLUTION APPROACH:
1. Understand what quantum state/operation is needed
2. Choose the minimal gate sequence
3. Verify the gates are available
4. Return ONLY the QASM code

Return the complete OpenQASM 2.0 circuit wrapped in code blocks.
Format:
```qasm
OPENQASM 2.0;
include "qelib1.inc";
[Your circuit here]
```

Remember: Simplicity and correctness first, optimization second."""

# =============================================================================
# ENHANCED QUANTUM CIRCUIT GENERATION (WITH OPTIMIZATION HINTS)
# =============================================================================

QUANTUM_CIRCUIT_OPTIMIZED_V2 = """You are an expert quantum circuit designer with deep knowledge of quantum gate theory and optimization.

TASK: Generate an OpenQASM 2.0 quantum circuit that solves the following problem.

PROBLEM: {problem_statement}

DESIGN REQUIREMENTS:
✓ Use exactly {min_qubits} qubit(s)
✓ Keep depth ≤ {max_depth if max_depth else "minimal"}
✓ Only use these gates: {required_gates}
✓ Do NOT use: {forbidden_gates if forbidden_gates else "none"}

CRITICAL RULES (must follow):
1. NO measurement operations unless explicitly required
2. NO extra qubits - use only what's needed
3. NO unnecessary gates - every gate serves a purpose
4. Prefer H, X, Z, CX over complex multi-qubit gates
5. Gate cancellations (e.g., X·X = I) are encouraged

OPTIMIZATION GUIDANCE:
- Minimize depth: Each qubit layer should have parallel operations where possible
- Minimize two-qubit gates: These are most expensive
- Look for identities: XX=I, ZZ=I, HZH=X, HXH=Z, etc.
- Consider what state you're creating, not just what gates to apply

SOLUTION CHECKLIST:
Before generating the circuit, think through:
1. What is the target quantum state? (e.g., |+⟩, |Φ+⟩, etc.)
2. What's the minimal gate sequence to create it?
3. Can any gates be combined or cancelled?
4. Is the depth truly minimal?

OUTPUT FORMAT:
Return ONLY the OpenQASM 2.0 code in a code block:

```qasm
OPENQASM 2.0;
include "qelib1.inc";
qreg q[{min_qubits}];
[Your gates here]
```

Do NOT include explanations, do NOT include measurements, do NOT use extra qubits."""

# =============================================================================
# SPECIALIZED PROMPTS FOR PROBLEM CATEGORIES
# =============================================================================

STATE_PREPARATION_PROMPT = """You are designing a quantum state preparation circuit.

PROBLEM: {problem_statement}

Your goal is to transform the initial state |0...0⟩ into the target quantum state.

TARGET STATE: {expected_states}

GATES AVAILABLE: {required_gates}

KEY INSIGHTS FOR STATE PREP:
- Hadamard (H) creates superposition: H|0⟩ = (|0⟩ + |1⟩)/√2
- Pauli-X flips: X|0⟩ = |1⟩, X|1⟩ = |0⟩
- Pauli-Z adds phase: Z|1⟩ = -|1⟩
- Phase flip: |−⟩ = (|0⟩ - |1⟩)/√2 requires X then H
- Bell states need H on first qubit, then CX

SOLUTION:
Return the minimal OpenQASM circuit:

```qasm
OPENQASM 2.0;
include "qelib1.inc";
qreg q[{min_qubits}];
[Your gates here]
```"""

ENTANGLEMENT_PROMPT = """You are designing an entanglement circuit.

PROBLEM: {problem_statement}

Your goal is to create entanglement between qubits.

TARGET: {expected_states}

ENTANGLEMENT FACTS:
- Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 requires: H on qubit 0, CX from 0→1
- Bell state |Φ-⟩ = (|00⟩ - |11⟩)/√2 requires: X on qubit 0, H on qubit 0, CX from 0→1
- GHZ state |GHZ⟩ = (|000⟩ + |111⟩)/√2 needs H on first, two CXs
- Entanglement requires multi-qubit gates (CX/CNOT)

SOLUTION:
Return the minimal OpenQASM circuit:

```qasm
OPENQASM 2.0;
include "qelib1.inc";
qreg q[{min_qubits}];
[Your gates here]
```"""

ALGORITHM_PROMPT = """You are implementing a quantum algorithm.

PROBLEM: {problem_statement}

ALGORITHM STRUCTURE:
{problem_statement}

KEY ALGORITHM COMPONENTS:
- Prepare superposition (usually with Hadamard)
- Apply oracle (function evaluation)
- Apply diffusion/phase flip (algorithm-specific)
- Measure result

SOLUTION:
Return the complete OpenQASM circuit:

```qasm
OPENQASM 2.0;
include "qelib1.inc";
qreg q[{min_qubits}];
[Your gates here]
```

Focus on correctness of the algorithm structure over minimal gate count."""

# =============================================================================
# GATE SYNTHESIS / DECOMPOSITION
# =============================================================================

GATE_SYNTHESIS_PROMPT = """You are decomposing a complex quantum gate into basic gates.

PROBLEM: {problem_statement}

TARGET GATE: {goal}

DECOMPOSITION FACTS:
- SWAP gate = 3 CX gates (CX a→b, CX b→a, CX a→b)
- CZ gate = H on target, CX, H on target
- Y gate = S·X·S†
- T gate = rotation by π/8 around Z-axis
- Rx(θ) = H·Rz(θ)·H (where applicable)

CONSTRAINTS:
- Only use: {required_gates}
- Avoid: {forbidden_gates if forbidden_gates else "none"}
- Minimize gate count and depth

SOLUTION:
Return the decomposed OpenQASM circuit:

```qasm
OPENQASM 2.0;
include "qelib1.inc";
qreg q[{min_qubits}];
[Your decomposition here]
```"""

# =============================================================================
# HELPER FUNCTION: FORMAT PROMPT FOR PROBLEM
# =============================================================================

def get_optimized_prompt(problem, use_advanced=True):
    """Generate optimized prompt for a problem.
    
    Args:
        problem: TestProblem instance
        use_advanced: Use advanced V2 prompt with optimization hints
        
    Returns:
        Formatted prompt string
    """
    template = QUANTUM_CIRCUIT_OPTIMIZED_V2 if use_advanced else QUANTUM_CIRCUIT_OPTIMIZED
    
    expected = problem.expected
    
    # Determine required and forbidden gates
    required_gates = expected.required_gates if expected.required_gates else ["h", "x", "z", "cx", "measure"]
    forbidden_gates = expected.forbidden_gates if expected.forbidden_gates else []
    
    # Format the prompt
    prompt = template.format(
        problem_statement=problem.prompt,
        min_qubits=expected.min_qubits,
        max_qubits=expected.max_qubits,
        max_depth=expected.max_depth or "minimal",
        required_gates=", ".join(required_gates),
        forbidden_gates=", ".join(forbidden_gates) if forbidden_gates else "none",
        expected_states=problem.expected.expected_states if hasattr(problem.expected, 'expected_states') else "N/A"
    )
    
    return prompt


def get_specialized_prompt(problem, use_advanced=True):
    """Generate specialized prompt based on problem category.
    
    Args:
        problem: TestProblem instance
        use_advanced: Use advanced optimization hints
        
    Returns:
        Formatted prompt string
    """
    from tests.test_problems import ProblemCategory
    
    category_prompts = {
        ProblemCategory.STATE_PREPARATION: STATE_PREPARATION_PROMPT,
        ProblemCategory.GATE_SYNTHESIS: GATE_SYNTHESIS_PROMPT,
        ProblemCategory.ALGORITHM: ALGORITHM_PROMPT,
        ProblemCategory.ERROR_CORRECTION: QUANTUM_CIRCUIT_OPTIMIZED_V2,
        ProblemCategory.OPTIMIZATION: QUANTUM_CIRCUIT_OPTIMIZED_V2,
    }
    
    template = category_prompts.get(problem.category, QUANTUM_CIRCUIT_OPTIMIZED_V2)
    
    expected = problem.expected
    required_gates = expected.required_gates if expected.required_gates else ["h", "x", "z", "cx"]
    forbidden_gates = expected.forbidden_gates if expected.forbidden_gates else []
    
    prompt = template.format(
        problem_statement=problem.prompt,
        goal=problem.name,
        min_qubits=expected.min_qubits,
        max_qubits=expected.max_qubits,
        max_depth=expected.max_depth or "minimal",
        required_gates=", ".join(required_gates),
        forbidden_gates=", ".join(forbidden_gates) if forbidden_gates else "none",
        expected_states=problem.expected.expected_states if hasattr(problem.expected, 'expected_states') else "N/A"
    )
    
    return prompt

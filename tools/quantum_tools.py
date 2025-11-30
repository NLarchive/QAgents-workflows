"""
Quantum Tools: MCP endpoint wrappers registered as tools.
All 23 MCP endpoints wrapped as callable tools for agents.
"""

from typing import Any, Dict, Optional
from .tool_registry import register_tool, ToolCategory, registry

# Import client lazily to avoid circular imports
def _get_client():
    from client import get_client
    return get_client()


# ===== CREATION TOOLS =====

@register_tool(
    name="create_from_template",
    description="Create a quantum circuit from a predefined template (bell_state, ghz, qft, grover, etc.)",
    category=ToolCategory.CREATION,
    parameters={
        "template": {"type": "string", "description": "Template name", "required": True},
        "num_qubits": {"type": "integer", "description": "Number of qubits", "required": False}
    },
    returns="QASM code of the created circuit"
)
def create_from_template(template: str, num_qubits: int = 2) -> Dict:
    response = _get_client().create_circuit_from_template(template, num_qubits)
    return {"success": response.success, "qasm": response.data, "error": response.error}


@register_tool(
    name="generate_random_circuit",
    description="Generate a random quantum circuit with specified parameters",
    category=ToolCategory.CREATION,
    parameters={
        "num_qubits": {"type": "integer", "description": "Number of qubits", "required": True},
        "depth": {"type": "integer", "description": "Circuit depth", "required": True},
        "gate_set": {"type": "string", "description": "Comma-separated gates (h,cx,rz)", "required": False}
    },
    returns="QASM code of the random circuit"
)
def generate_random_circuit(num_qubits: int, depth: int, gate_set: str = "h,cx,rz") -> Dict:
    response = _get_client().generate_random_circuit(num_qubits, depth, gate_set)
    return {"success": response.success, "qasm": response.data, "error": response.error}


@register_tool(
    name="generate_from_description",
    description="Generate a circuit from natural language description",
    category=ToolCategory.CREATION,
    parameters={
        "description": {"type": "string", "description": "Natural language description of the circuit", "required": True}
    },
    returns="QASM code of the generated circuit"
)
def generate_from_description(description: str) -> Dict:
    response = _get_client().generate_circuit_from_description(description)
    return {"success": response.success, "qasm": response.data, "error": response.error}


# ===== ANALYSIS TOOLS =====

@register_tool(
    name="parse_qasm",
    description="Parse OpenQASM code and extract circuit structure",
    category=ToolCategory.ANALYSIS,
    parameters={
        "qasm": {"type": "string", "description": "OpenQASM code", "required": True}
    },
    returns="Parsed circuit structure with gates, qubits, etc."
)
def parse_qasm(qasm: str) -> Dict:
    response = _get_client().parse_qasm(qasm)
    return {"success": response.success, "structure": response.data, "error": response.error}


@register_tool(
    name="analyze_circuit",
    description="Analyze circuit properties: depth, gate count, qubit usage",
    category=ToolCategory.ANALYSIS,
    parameters={
        "qasm": {"type": "string", "description": "OpenQASM code", "required": True}
    },
    returns="Circuit analysis with depth, gate counts, etc."
)
def analyze_circuit(qasm: str) -> Dict:
    response = _get_client().analyze_circuit(qasm)
    return {"success": response.success, "analysis": response.data, "error": response.error}


@register_tool(
    name="get_circuit_depth",
    description="Get the depth of a quantum circuit",
    category=ToolCategory.ANALYSIS,
    parameters={
        "qasm": {"type": "string", "description": "OpenQASM code", "required": True}
    },
    returns="Integer depth value"
)
def get_circuit_depth(qasm: str) -> Dict:
    response = _get_client().get_circuit_depth(qasm)
    return {"success": response.success, "depth": response.data, "error": response.error}


# ===== VALIDATION TOOLS =====

@register_tool(
    name="validate_syntax",
    description="Validate QASM syntax for correctness",
    category=ToolCategory.VALIDATION,
    parameters={
        "qasm": {"type": "string", "description": "OpenQASM code", "required": True}
    },
    returns="Validation result with any syntax errors"
)
def validate_syntax(qasm: str) -> Dict:
    response = _get_client().validate_syntax(qasm)
    return {"success": response.success, "valid": response.data, "error": response.error}


@register_tool(
    name="check_connectivity",
    description="Check if circuit respects hardware qubit connectivity",
    category=ToolCategory.VALIDATION,
    parameters={
        "qasm": {"type": "string", "description": "OpenQASM code", "required": True},
        "hardware": {"type": "string", "description": "Hardware profile (ibm_eagle, ionq_aria, rigetti_aspen)", "required": False}
    },
    returns="Connectivity check result"
)
def check_connectivity(qasm: str, hardware: str = "ibm_eagle") -> Dict:
    response = _get_client().check_connectivity(qasm, hardware)
    return {"success": response.success, "result": response.data, "error": response.error}


@register_tool(
    name="verify_unitary",
    description="Verify that circuit produces a valid unitary matrix",
    category=ToolCategory.VALIDATION,
    parameters={
        "qasm": {"type": "string", "description": "OpenQASM code", "required": True}
    },
    returns="Unitary verification result"
)
def verify_unitary(qasm: str) -> Dict:
    response = _get_client().verify_unitary(qasm)
    return {"success": response.success, "result": response.data, "error": response.error}


# ===== SIMULATION TOOLS =====

@register_tool(
    name="simulate_circuit",
    description="Simulate circuit execution and get measurement results",
    category=ToolCategory.SIMULATION,
    parameters={
        "qasm": {"type": "string", "description": "OpenQASM code", "required": True},
        "shots": {"type": "integer", "description": "Number of measurement shots", "required": False}
    },
    returns="Measurement results with counts"
)
def simulate_circuit(qasm: str, shots: int = 1024) -> Dict:
    response = _get_client().simulate_circuit(qasm, shots)
    return {"success": response.success, "results": response.data, "error": response.error}


@register_tool(
    name="get_statevector",
    description="Get the statevector of a circuit (no measurement)",
    category=ToolCategory.SIMULATION,
    parameters={
        "qasm": {"type": "string", "description": "OpenQASM code", "required": True}
    },
    returns="Statevector as complex amplitudes"
)
def get_statevector(qasm: str) -> Dict:
    response = _get_client().get_statevector(qasm)
    return {"success": response.success, "statevector": response.data, "error": response.error}


@register_tool(
    name="get_probabilities",
    description="Get probability distribution from circuit",
    category=ToolCategory.SIMULATION,
    parameters={
        "qasm": {"type": "string", "description": "OpenQASM code", "required": True}
    },
    returns="Probability distribution over computational basis states"
)
def get_probabilities(qasm: str) -> Dict:
    response = _get_client().get_probabilities(qasm)
    return {"success": response.success, "probabilities": response.data, "error": response.error}


# ===== SCORING TOOLS =====

@register_tool(
    name="calculate_complexity",
    description="Calculate circuit complexity score (lower is better)",
    category=ToolCategory.SCORING,
    parameters={
        "qasm": {"type": "string", "description": "OpenQASM code", "required": True}
    },
    returns="Complexity score and breakdown"
)
def calculate_complexity(qasm: str) -> Dict:
    response = _get_client().calculate_complexity_score(qasm)
    return {"success": response.success, "score": response.data, "error": response.error}


@register_tool(
    name="calculate_hardware_fitness",
    description="Calculate how well circuit fits target hardware",
    category=ToolCategory.SCORING,
    parameters={
        "qasm": {"type": "string", "description": "OpenQASM code", "required": True},
        "hardware": {"type": "string", "description": "Hardware profile", "required": False}
    },
    returns="Hardware fitness score (higher is better)"
)
def calculate_hardware_fitness(qasm: str, hardware: str = "ibm_eagle") -> Dict:
    response = _get_client().calculate_hardware_fitness(qasm, hardware)
    return {"success": response.success, "score": response.data, "error": response.error}


@register_tool(
    name="calculate_expressibility",
    description="Calculate circuit expressibility (ability to explore state space)",
    category=ToolCategory.SCORING,
    parameters={
        "qasm": {"type": "string", "description": "OpenQASM code", "required": True}
    },
    returns="Expressibility score"
)
def calculate_expressibility(qasm: str) -> Dict:
    response = _get_client().calculate_expressibility(qasm)
    return {"success": response.success, "score": response.data, "error": response.error}


# ===== RESOURCE TOOLS =====

@register_tool(
    name="estimate_resources",
    description="Estimate resource requirements (qubits, gates, depth)",
    category=ToolCategory.RESOURCE,
    parameters={
        "qasm": {"type": "string", "description": "OpenQASM code", "required": True}
    },
    returns="Resource estimation breakdown"
)
def estimate_resources(qasm: str) -> Dict:
    response = _get_client().estimate_resources(qasm)
    return {"success": response.success, "resources": response.data, "error": response.error}


@register_tool(
    name="estimate_noise",
    description="Estimate noise impact on circuit execution",
    category=ToolCategory.RESOURCE,
    parameters={
        "qasm": {"type": "string", "description": "OpenQASM code", "required": True},
        "hardware": {"type": "string", "description": "Hardware profile", "required": False}
    },
    returns="Noise estimation"
)
def estimate_noise(qasm: str, hardware: str = "ibm_eagle") -> Dict:
    response = _get_client().estimate_noise(qasm, hardware)
    return {"success": response.success, "noise": response.data, "error": response.error}


# ===== COMPOSITION TOOLS =====

@register_tool(
    name="compose_circuits",
    description="Compose two circuits sequentially",
    category=ToolCategory.COMPOSITION,
    parameters={
        "qasm1": {"type": "string", "description": "First circuit QASM", "required": True},
        "qasm2": {"type": "string", "description": "Second circuit QASM", "required": True},
        "qubit_mapping": {"type": "string", "description": "Qubit mapping (e.g., '0:1,1:0')", "required": False}
    },
    returns="Composed circuit QASM"
)
def compose_circuits(qasm1: str, qasm2: str, qubit_mapping: str = "") -> Dict:
    response = _get_client().compose_circuits(qasm1, qasm2, qubit_mapping)
    return {"success": response.success, "qasm": response.data, "error": response.error}


@register_tool(
    name="generate_inverse",
    description="Generate the inverse (adjoint) of a circuit",
    category=ToolCategory.COMPOSITION,
    parameters={
        "qasm": {"type": "string", "description": "OpenQASM code", "required": True}
    },
    returns="Inverse circuit QASM"
)
def generate_inverse(qasm: str) -> Dict:
    response = _get_client().generate_inverse_circuit(qasm)
    return {"success": response.success, "qasm": response.data, "error": response.error}


@register_tool(
    name="tensor_circuits",
    description="Create tensor product of two circuits (parallel composition)",
    category=ToolCategory.COMPOSITION,
    parameters={
        "qasm1": {"type": "string", "description": "First circuit QASM", "required": True},
        "qasm2": {"type": "string", "description": "Second circuit QASM", "required": True}
    },
    returns="Tensored circuit QASM"
)
def tensor_circuits(qasm1: str, qasm2: str) -> Dict:
    response = _get_client().tensor_circuits(qasm1, qasm2)
    return {"success": response.success, "qasm": response.data, "error": response.error}


@register_tool(
    name="repeat_circuit",
    description="Repeat a circuit n times",
    category=ToolCategory.COMPOSITION,
    parameters={
        "qasm": {"type": "string", "description": "OpenQASM code", "required": True},
        "n": {"type": "integer", "description": "Number of repetitions", "required": True}
    },
    returns="Repeated circuit QASM"
)
def repeat_circuit(qasm: str, n: int) -> Dict:
    response = _get_client().repeat_circuit(qasm, n)
    return {"success": response.success, "qasm": response.data, "error": response.error}


# ===== UTILITY FUNCTIONS =====

def get_all_tools():
    """Get all registered tools."""
    return registry.get_all()

def get_tools_by_category(category: ToolCategory):
    """Get tools by category."""
    return registry.get_by_category(category)

def invoke_tool(name: str, **kwargs):
    """Invoke a tool by name."""
    return registry.invoke(name, **kwargs)

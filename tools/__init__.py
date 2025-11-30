"""Tools module: MCP endpoint wrappers as callable tools."""

from .tool_registry import (
    ToolDefinition,
    ToolCategory,
    ToolRegistry,
    registry,
    register_tool
)

from .quantum_tools import (
    get_all_tools,
    get_tools_by_category,
    invoke_tool,
    # Creation tools
    create_from_template,
    generate_random_circuit,
    generate_from_description,
    # Analysis tools
    parse_qasm,
    analyze_circuit,
    get_circuit_depth,
    # Validation tools
    validate_syntax,
    check_connectivity,
    verify_unitary,
    # Simulation tools
    simulate_circuit,
    get_statevector,
    get_probabilities,
    # Scoring tools
    calculate_complexity,
    calculate_hardware_fitness,
    calculate_expressibility,
    # Resource tools
    estimate_resources,
    estimate_noise,
    # Composition tools
    compose_circuits,
    generate_inverse,
    tensor_circuits,
    repeat_circuit
)

__all__ = [
    "ToolDefinition",
    "ToolCategory", 
    "ToolRegistry",
    "registry",
    "register_tool",
    "get_all_tools",
    "get_tools_by_category",
    "invoke_tool"
]

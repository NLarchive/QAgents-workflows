# Path: QAgents-workflos/test_quality_analyzer.py
# Description: Test the circuit quality analyzer
"""Test that quality analyzer works with MCP endpoints."""

from tests.circuit_quality_analyzer import CircuitQualityAnalyzer, get_analyzer

def test_analyzer():
    analyzer = get_analyzer()
    
    # Test with a Bell state circuit
    test_qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""

    print("Analyzing Bell state circuit...")
    print("-" * 40)
    
    result = analyzer.analyze_circuit(test_qasm)
    
    print(f"Syntax Valid: {result.syntax_valid}")
    print(f"Depth: {result.depth}")
    print(f"Gate Count: {result.gate_count}")
    print(f"CX Count: {result.cx_count}")
    print(f"Single Qubit Count: {result.single_qubit_count}")
    print(f"Hardware Fitness: {result.hardware_fitness}")
    print(f"Complexity Score: {result.complexity_score}")
    print(f"State Correctness: {result.state_correctness}")
    print(f"Noise Estimate: {result.noise_estimate}")
    print(f"Probabilities: {result.probabilities}")
    
    if result.errors:
        print(f"\nErrors/Warnings:")
        for err in result.errors:
            print(f"  - {err}")

if __name__ == "__main__":
    test_analyzer()

# Path: QAgents-workflos/tests/test_mcp_client.py
# Relations: Tests client/mcp_client.py
# Description: Comprehensive tests for MCP client with Gradio and fallback implementations

"""
Test suite for MCP client functionality.
Tests both Gradio-based endpoints and local fallback implementations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.mcp_client import get_client, MCPClient, QASMLocalAnalyzer

# Sample QASM for testing
BELL_STATE_QASM = '''OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;'''


def test_health_check():
    """Test server health check."""
    client = get_client()
    result = client.health_check()
    print(f"Health Check: {'OK' if result else 'FAILED'}")
    return result


def test_create_circuit():
    """Test circuit creation from template (uses Gradio)."""
    client = get_client()
    result = client.create_circuit_from_template('bell_state', 2)
    
    print(f"Create Circuit:")
    print(f"  Success: {result.success}")
    print(f"  Endpoint: {result.endpoint}")
    print(f"  Time: {result.execution_time_ms:.2f}ms")
    if result.success and result.data:
        print(f"  Data preview: {str(result.data)[:80]}...")
    return result.success


def test_analyze_circuit():
    """Test circuit analysis (uses fallback)."""
    client = get_client()
    result = client.analyze_circuit(BELL_STATE_QASM)
    
    print(f"Analyze Circuit:")
    print(f"  Success: {result.success}")
    print(f"  Is Fallback: {result.is_fallback}")
    if result.success:
        print(f"  Depth: {result.data.get('depth')}")
        print(f"  Gate Count: {result.data.get('gate_count')}")
        print(f"  Two-qubit Gates: {result.data.get('two_qubit_gates')}")
    return result.success


def test_validate_syntax():
    """Test syntax validation (uses Gradio)."""
    client = get_client()
    result = client.validate_syntax(BELL_STATE_QASM)
    
    print(f"Validate Syntax:")
    print(f"  Success: {result.success}")
    print(f"  Endpoint: {result.endpoint}")
    print(f"  Time: {result.execution_time_ms:.2f}ms")
    return result.success


def test_simulate_circuit():
    """Test circuit simulation (uses Gradio)."""
    client = get_client()
    result = client.simulate_circuit(BELL_STATE_QASM, shots=100)
    
    print(f"Simulate Circuit:")
    print(f"  Success: {result.success}")
    print(f"  Endpoint: {result.endpoint}")
    print(f"  Time: {result.execution_time_ms:.2f}ms")
    if result.success and result.data:
        print(f"  Data preview: {str(result.data)[:80]}...")
    return result.success


def test_complexity_score():
    """Test complexity scoring (uses Gradio or fallback)."""
    client = get_client()
    result = client.calculate_complexity_score(BELL_STATE_QASM)
    
    print(f"Complexity Score:")
    print(f"  Success: {result.success}")
    print(f"  Is Fallback: {result.is_fallback}")
    if result.success and result.data:
        if isinstance(result.data, dict):
            print(f"  Score: {result.data.get('complexity_score', 'N/A')}")
    return result.success


def test_estimate_noise():
    """Test noise estimation (uses fallback)."""
    client = get_client()
    result = client.estimate_noise(BELL_STATE_QASM, hardware='ibm_brisbane')
    
    print(f"Estimate Noise:")
    print(f"  Success: {result.success}")
    print(f"  Is Fallback: {result.is_fallback}")
    if result.success:
        print(f"  Fidelity: {result.data.get('estimated_fidelity')}")
        print(f"  Total Error: {result.data.get('total_error_probability')}")
    return result.success


def test_local_analyzer():
    """Test QASMLocalAnalyzer directly."""
    analyzer = QASMLocalAnalyzer()
    
    # Parse
    parsed = analyzer.parse_qasm(BELL_STATE_QASM)
    print(f"Local Parser:")
    print(f"  Qubits: {parsed['num_qubits']}")
    print(f"  Gates: {len(parsed['gates'])}")
    
    # Analyze
    analysis = analyzer.analyze_circuit(BELL_STATE_QASM)
    print(f"Local Analyzer:")
    print(f"  Depth: {analysis['depth']}")
    print(f"  Gate breakdown: {analysis['gate_breakdown']}")
    
    # Complexity
    complexity = analyzer.calculate_complexity(BELL_STATE_QASM)
    print(f"Local Complexity:")
    print(f"  Score: {complexity['complexity_score']}")
    
    return True


def run_all_tests():
    """Run all MCP client tests."""
    print("=" * 50)
    print("MCP Client Test Suite")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("Create Circuit", test_create_circuit),
        ("Analyze Circuit", test_analyze_circuit),
        ("Validate Syntax", test_validate_syntax),
        ("Simulate Circuit", test_simulate_circuit),
        ("Complexity Score", test_complexity_score),
        ("Estimate Noise", test_estimate_noise),
        ("Local Analyzer", test_local_analyzer),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    passed = sum(1 for _, p in results if p)
    print(f"Passed: {passed}/{len(results)}")
    for name, p in results:
        status = "✓" if p else "✗"
        print(f"  {status} {name}")
    
    return all(p for _, p in results)


if __name__ == "__main__":
    run_all_tests()

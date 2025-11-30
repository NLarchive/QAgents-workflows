# Path: QAgents-workflos/tests/circuit_quality_analyzer.py
# Relations: Uses client/mcp_client.py for MCP calls, database/circuit_quality_db.py for storage
# Description: Analyzes circuit quality using MCP endpoints
#              Extracts: depth, gate_count, cx_count, hardware_fitness, validation, simulation
#              Returns QualityMetrics for storage in database

"""
Circuit Quality Analyzer: Use MCP endpoints to measure circuit quality.
This module connects to the MCP server and extracts quality metrics.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result from analyzing a circuit."""
    depth: int = 0
    gate_count: int = 0
    cx_count: int = 0
    single_qubit_count: int = 0
    hardware_fitness: float = 0.0
    syntax_valid: bool = False
    complexity_score: float = 0.0
    state_correctness: float = 0.0
    noise_estimate: float = 0.0
    probabilities: Dict[str, float] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.probabilities is None:
            self.probabilities = {}
        if self.errors is None:
            self.errors = []


class CircuitQualityAnalyzer:
    """
    Analyzes circuit quality using MCP endpoints.
    Connects to the running MCP server to get quality metrics.
    """
    
    def __init__(self, mcp_url: str = "http://127.0.0.1:7861"):
        self.mcp_url = mcp_url
        self._client = None
    
    def _get_client(self):
        """Get or create MCP client."""
        if self._client is None:
            try:
                from client import get_client
                self._client = get_client(self.mcp_url)
            except Exception as e:
                logger.error(f"Failed to get MCP client: {e}")
                return None
        return self._client
    
    def _extract_value(self, result: Any, keys: List[str], default: Any = 0) -> Any:
        """Safely extract value from nested result."""
        if result is None:
            return default
        
        if isinstance(result, (int, float, bool)):
            return result
        
        if isinstance(result, list):
            return result[0] if result else default
        
        if isinstance(result, dict):
            for key in keys:
                if key in result:
                    val = result[key]
                    if isinstance(val, (int, float)):
                        return val
                    elif isinstance(val, dict):
                        # Try common nested keys
                        for subkey in ['value', 'score', 'depth', 'count', 'result']:
                            if subkey in val:
                                return val[subkey]
                    elif isinstance(val, list):
                        return val[0] if val else default
                    return val
            # Try first value in dict
            for v in result.values():
                if isinstance(v, (int, float)):
                    return v
        
        return default
    
    def analyze_circuit(self, qasm_code: str, expected_states: Dict[str, float] = None) -> AnalysisResult:
        """
        Analyze a circuit using MCP endpoints.
        
        Args:
            qasm_code: The QASM code to analyze
            expected_states: Expected probability distribution for correctness check
        
        Returns:
            AnalysisResult with all quality metrics
        """
        result = AnalysisResult()
        
        if not qasm_code or not qasm_code.strip():
            result.errors.append("Empty QASM code")
            return result
        
        client = self._get_client()
        if client is None:
            # Fallback to local analysis
            return self._analyze_locally(qasm_code, expected_states)
        
        # 1. Validate syntax
        try:
            resp = client.validate_syntax(qasm_code)
            if resp.success:
                valid = resp.data
                if isinstance(valid, dict):
                    result.syntax_valid = valid.get('valid', False) or valid.get('is_valid', False)
                elif isinstance(valid, bool):
                    result.syntax_valid = valid
                elif isinstance(valid, list):
                    result.syntax_valid = "valid" in str(valid).lower()
                else:
                    result.syntax_valid = bool(valid)
            else:
                result.errors.append(f"Validation error: {resp.error}")
        except Exception as e:
            result.errors.append(f"Validation failed: {e}")
            # Still try to parse locally
            result.syntax_valid = "OPENQASM" in qasm_code and "qreg" in qasm_code
        
        # 2. Analyze circuit structure
        try:
            resp = client.analyze_circuit(qasm_code)
            if resp.success and resp.data:
                data = resp.data
                if isinstance(data, dict):
                    result.depth = self._extract_value(data, ['depth', 'circuit_depth'], 0)
                    result.gate_count = self._extract_value(data, ['gate_count', 'gates', 'num_gates', 'total_gates'], 0)
                    result.cx_count = self._extract_value(data, ['cx_count', 'cnot_count', 'two_qubit_gates'], 0)
                    result.single_qubit_count = self._extract_value(data, ['single_qubit_count', 'single_qubit_gates', 'one_qubit_gates'], 0)
        except Exception as e:
            result.errors.append(f"Analysis failed: {e}")
            # Fallback to local parsing
            local = self._parse_qasm_locally(qasm_code)
            result.depth = local.get('depth', 0)
            result.gate_count = local.get('gate_count', 0)
            result.cx_count = local.get('cx_count', 0)
            result.single_qubit_count = local.get('single_qubit_count', 0)
        
        # 3. Get circuit depth if not already set
        if result.depth == 0:
            try:
                resp = client.get_circuit_depth(qasm_code)
                if resp.success:
                    result.depth = self._extract_value(resp.data, ['depth', 'value'], 0)
            except Exception as e:
                result.errors.append(f"Depth check failed: {e}")
        
        # 4. Calculate hardware fitness
        try:
            resp = client.calculate_hardware_fitness(qasm_code, "ibm_brisbane")
            if resp.success:
                result.hardware_fitness = self._extract_value(resp.data, 
                    ['fitness', 'fitness_score', 'hardware_fitness', 'score'], 0.0)
                if result.hardware_fitness > 1.0:
                    result.hardware_fitness = result.hardware_fitness / 100.0
        except Exception as e:
            result.errors.append(f"Hardware fitness failed: {e}")
        
        # 5. Calculate complexity
        try:
            resp = client.calculate_complexity_score(qasm_code)
            if resp.success:
                result.complexity_score = self._extract_value(resp.data,
                    ['complexity', 'complexity_score', 'score', 'total'], 0.0)
        except Exception as e:
            result.errors.append(f"Complexity check failed: {e}")
        
        # 6. Get probabilities and check correctness
        try:
            resp = client.get_probabilities(qasm_code)
            if resp.success and resp.data:
                probs = resp.data
                if isinstance(probs, dict):
                    result.probabilities = probs
                    if expected_states:
                        result.state_correctness = self._check_correctness(probs, expected_states)
                    else:
                        # No expected states - assume 100% if circuit runs
                        result.state_correctness = 1.0
        except Exception as e:
            result.errors.append(f"Probability check failed: {e}")
            if expected_states is None:
                result.state_correctness = 0.8  # Partial credit if other metrics pass
        
        # 7. Estimate noise
        try:
            resp = client.estimate_noise(qasm_code, "ibm_brisbane")
            if resp.success:
                result.noise_estimate = self._extract_value(resp.data,
                    ['noise', 'noise_estimate', 'error_rate', 'fidelity'], 0.0)
        except Exception as e:
            result.errors.append(f"Noise estimation failed: {e}")
        
        return result
    
    def _analyze_locally(self, qasm_code: str, expected_states: Dict[str, float] = None) -> AnalysisResult:
        """Fallback local analysis when MCP is unavailable."""
        result = AnalysisResult()
        
        # Basic syntax check
        result.syntax_valid = "OPENQASM" in qasm_code and "qreg" in qasm_code
        
        # Parse gates
        local = self._parse_qasm_locally(qasm_code)
        result.depth = local.get('depth', 0)
        result.gate_count = local.get('gate_count', 0)
        result.cx_count = local.get('cx_count', 0)
        result.single_qubit_count = local.get('single_qubit_count', 0)
        
        # Estimate hardware fitness based on structure
        if result.gate_count > 0:
            # Penalize high CX ratio
            cx_ratio = result.cx_count / result.gate_count
            result.hardware_fitness = max(0.0, 1.0 - cx_ratio * 0.5)
        
        # Complexity estimate
        result.complexity_score = result.depth + result.cx_count * 2
        
        # State correctness if syntax valid
        if result.syntax_valid:
            result.state_correctness = 0.7  # Partial credit
        
        result.errors.append("Used local fallback analysis")
        return result
    
    def _parse_qasm_locally(self, qasm_code: str) -> Dict[str, int]:
        """Parse QASM locally to extract gate counts."""
        result = {
            'depth': 0,
            'gate_count': 0,
            'cx_count': 0,
            'single_qubit_count': 0
        }
        
        lines = qasm_code.strip().split('\n')
        gate_depth_map = {}  # qubit -> current depth
        
        single_qubit_gates = ['h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg', 'rx', 'ry', 'rz', 'u1', 'u2', 'u3']
        two_qubit_gates = ['cx', 'cz', 'swap', 'cp', 'crz', 'cnot']
        
        for line in lines:
            line = line.strip().lower()
            if not line or line.startswith('//') or line.startswith('openqasm') or line.startswith('include'):
                continue
            if line.startswith('qreg') or line.startswith('creg') or line.startswith('measure') or line.startswith('barrier'):
                continue
            
            # Check for gates
            for gate in single_qubit_gates:
                if line.startswith(gate + ' ') or line.startswith(gate + '('):
                    result['single_qubit_count'] += 1
                    result['gate_count'] += 1
                    # Extract qubit
                    match = re.search(r'q\[(\d+)\]', line)
                    if match:
                        q = int(match.group(1))
                        gate_depth_map[q] = gate_depth_map.get(q, 0) + 1
                    break
            
            for gate in two_qubit_gates:
                if line.startswith(gate + ' '):
                    result['cx_count'] += 1
                    result['gate_count'] += 1
                    # Extract qubits
                    matches = re.findall(r'q\[(\d+)\]', line)
                    if matches:
                        for q in matches:
                            q = int(q)
                            gate_depth_map[q] = gate_depth_map.get(q, 0) + 1
                    break
        
        if gate_depth_map:
            result['depth'] = max(gate_depth_map.values())
        
        return result
    
    def _check_correctness(self, actual: Dict[str, float], expected: Dict[str, float]) -> float:
        """Check how close actual probabilities are to expected."""
        if not expected:
            return 1.0
        
        total_error = 0.0
        for state, exp_prob in expected.items():
            act_prob = actual.get(state, 0.0)
            total_error += abs(exp_prob - act_prob)
        
        # Also check for unexpected states
        for state, act_prob in actual.items():
            if state not in expected and act_prob > 0.01:
                total_error += act_prob
        
        # Normalize (max error = 2.0)
        correctness = max(0.0, 1.0 - total_error / 2.0)
        return correctness
    
    def compare_circuits(self, qasm1: str, qasm2: str) -> Dict[str, Any]:
        """Compare two circuits and return quality differences."""
        result1 = self.analyze_circuit(qasm1)
        result2 = self.analyze_circuit(qasm2)
        
        return {
            "circuit1": {
                "depth": result1.depth,
                "gate_count": result1.gate_count,
                "cx_count": result1.cx_count,
                "hardware_fitness": result1.hardware_fitness,
                "syntax_valid": result1.syntax_valid
            },
            "circuit2": {
                "depth": result2.depth,
                "gate_count": result2.gate_count,
                "cx_count": result2.cx_count,
                "hardware_fitness": result2.hardware_fitness,
                "syntax_valid": result2.syntax_valid
            },
            "comparison": {
                "depth_diff": result2.depth - result1.depth,
                "gate_diff": result2.gate_count - result1.gate_count,
                "cx_diff": result2.cx_count - result1.cx_count,
                "fitness_diff": result2.hardware_fitness - result1.hardware_fitness,
                "circuit1_better": result1.depth < result2.depth or result1.hardware_fitness > result2.hardware_fitness
            }
        }


# Module-level singleton
_analyzer: Optional[CircuitQualityAnalyzer] = None

def get_analyzer(mcp_url: str = "http://127.0.0.1:7861") -> CircuitQualityAnalyzer:
    """Get or create the quality analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = CircuitQualityAnalyzer(mcp_url)
    return _analyzer

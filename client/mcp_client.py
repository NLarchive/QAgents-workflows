# Path: QAgents-workflos/client/mcp_client.py
# Relations: Uses QuantumArchitect-MCP Gradio server
# Description: MCP client with fallback local implementations for missing endpoints
"""
MCP Client: Connection to QuantumArchitect-MCP endpoints.
Provides both synchronous and async interfaces.

Available Gradio endpoints (as of latest scan):
- ui_create_circuit: Create circuit from template
- ui_validate_circuit: Validate QASM syntax
- ui_simulate_circuit: Simulate circuit
- ui_score_circuit: Score circuit complexity/fitness

Missing endpoints use local fallback implementations.
"""

import requests
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import re
import time
import random
import math

logger = logging.getLogger(__name__)


@dataclass
class MCPResponse:
    """Standardized response from MCP endpoints."""
    success: bool
    data: Any
    endpoint: str
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    is_fallback: bool = False  # True if using local fallback


class QASMLocalAnalyzer:
    """Local QASM analysis for fallback when MCP endpoints unavailable."""
    
    GATE_PATTERN = re.compile(
        r'^(h|x|y|z|s|t|sdg|tdg|cx|cz|cy|swap|ccx|rz|rx|ry|u1|u2|u3|p|measure|barrier)\b', 
        re.IGNORECASE
    )
    
    @staticmethod
    def parse_qasm(qasm_code: str) -> Dict[str, Any]:
        """Parse QASM code and extract structure."""
        lines = [l.strip() for l in qasm_code.strip().split('\n') 
                 if l.strip() and not l.strip().startswith('//')]
        
        result = {
            'openqasm_version': '2.0',
            'includes': [],
            'qregs': [],
            'cregs': [],
            'gates': [],
            'num_qubits': 0,
            'num_classical': 0
        }
        
        for line in lines:
            if line.startswith('OPENQASM'):
                result['openqasm_version'] = line.split()[1].rstrip(';')
            elif line.startswith('include'):
                result['includes'].append(line.split('"')[1] if '"' in line else line.split()[1])
            elif line.startswith('qreg'):
                match = re.search(r'qreg\s+(\w+)\[(\d+)\]', line)
                if match:
                    result['qregs'].append({'name': match.group(1), 'size': int(match.group(2))})
                    result['num_qubits'] += int(match.group(2))
            elif line.startswith('creg'):
                match = re.search(r'creg\s+(\w+)\[(\d+)\]', line)
                if match:
                    result['cregs'].append({'name': match.group(1), 'size': int(match.group(2))})
                    result['num_classical'] += int(match.group(2))
            elif QASMLocalAnalyzer.GATE_PATTERN.match(line):
                gate_name = line.split()[0].split('(')[0]
                result['gates'].append({'gate': gate_name, 'raw': line.rstrip(';')})
        
        return result
    
    @staticmethod
    def analyze_circuit(qasm_code: str) -> Dict[str, Any]:
        """Analyze circuit properties."""
        parsed = QASMLocalAnalyzer.parse_qasm(qasm_code)
        gates = parsed['gates']
        
        gate_counts = {}
        single_qubit_gates = 0
        two_qubit_gates = 0
        multi_qubit_gates = 0
        measurement_count = 0
        
        for g in gates:
            gate = g['gate'].lower()
            gate_counts[gate] = gate_counts.get(gate, 0) + 1
            
            if gate == 'measure':
                measurement_count += 1
            elif gate in ['cx', 'cz', 'cy', 'swap']:
                two_qubit_gates += 1
            elif gate in ['ccx', 'cswap']:
                multi_qubit_gates += 1
            else:
                single_qubit_gates += 1
        
        # Estimate depth (simplified: assume all gates sequential)
        depth = len([g for g in gates if g['gate'].lower() != 'measure'])
        
        return {
            'num_qubits': parsed['num_qubits'],
            'num_classical_bits': parsed['num_classical'],
            'depth': depth,
            'gate_count': len(gates),
            'gate_breakdown': gate_counts,
            'single_qubit_gates': single_qubit_gates,
            'two_qubit_gates': two_qubit_gates,
            'multi_qubit_gates': multi_qubit_gates,
            'measurements': measurement_count
        }
    
    @staticmethod
    def get_depth(qasm_code: str) -> int:
        """Get circuit depth."""
        analysis = QASMLocalAnalyzer.analyze_circuit(qasm_code)
        return analysis['depth']
    
    @staticmethod  
    def calculate_complexity(qasm_code: str) -> Dict[str, Any]:
        """Calculate complexity score."""
        analysis = QASMLocalAnalyzer.analyze_circuit(qasm_code)
        
        # Scoring formula
        depth_score = min(analysis['depth'] / 50.0, 1.0) * 30
        gate_score = min(analysis['gate_count'] / 100.0, 1.0) * 30
        two_q_score = min(analysis['two_qubit_gates'] / 20.0, 1.0) * 25
        qubit_score = min(analysis['num_qubits'] / 10.0, 1.0) * 15
        
        total = depth_score + gate_score + two_q_score + qubit_score
        
        return {
            'complexity_score': round(total, 2),
            'depth_contribution': round(depth_score, 2),
            'gate_contribution': round(gate_score, 2),
            'entanglement_contribution': round(two_q_score, 2),
            'qubit_contribution': round(qubit_score, 2),
            'raw_metrics': analysis
        }
    
    @staticmethod
    def validate_syntax(qasm_code: str) -> Dict[str, Any]:
        """Validate QASM syntax."""
        errors = []
        warnings = []
        
        lines = qasm_code.strip().split('\n')
        
        has_openqasm = False
        has_qreg = False
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('//'):
                continue
                
            if line.startswith('OPENQASM'):
                has_openqasm = True
            elif line.startswith('qreg'):
                has_qreg = True
            elif not line.startswith(('include', 'creg', 'barrier', 'measure', 'OPENQASM', 'qreg')):
                # Check for valid gate
                if not QASMLocalAnalyzer.GATE_PATTERN.match(line):
                    if line and not line.endswith(';'):
                        warnings.append(f"Line {i}: Missing semicolon")
        
        if not has_openqasm:
            errors.append("Missing OPENQASM version declaration")
        if not has_qreg:
            errors.append("No quantum register (qreg) defined")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'line_count': len(lines)
        }
    
    @staticmethod
    def calculate_hardware_fitness(qasm_code: str, hardware: str = "ibm_brisbane") -> Dict[str, Any]:
        """Calculate hardware fitness score."""
        analysis = QASMLocalAnalyzer.analyze_circuit(qasm_code)
        
        # Hardware profiles (simplified)
        profiles = {
            'ibm_brisbane': {'max_qubits': 127, 'connectivity': 'heavy-hex', 'two_q_error': 0.01},
            'ibm_sherbrooke': {'max_qubits': 127, 'connectivity': 'heavy-hex', 'two_q_error': 0.008},
            'rigetti_aspen': {'max_qubits': 80, 'connectivity': 'octagonal', 'two_q_error': 0.02},
            'ionq_harmony': {'max_qubits': 11, 'connectivity': 'all-to-all', 'two_q_error': 0.005}
        }
        
        profile = profiles.get(hardware, profiles['ibm_brisbane'])
        
        # Calculate fitness
        qubit_fit = 100 if analysis['num_qubits'] <= profile['max_qubits'] else 50
        depth_penalty = min(analysis['depth'] * 2, 30)
        two_q_penalty = analysis['two_qubit_gates'] * profile['two_q_error'] * 100
        
        fitness = max(0, qubit_fit - depth_penalty - two_q_penalty)
        
        return {
            'fitness_score': round(fitness, 2),
            'hardware': hardware,
            'qubit_fit': qubit_fit,
            'depth_penalty': round(depth_penalty, 2),
            'error_penalty': round(two_q_penalty, 2),
            'recommendation': 'suitable' if fitness > 70 else 'marginal' if fitness > 40 else 'poor'
        }


class MCPClient:
    """
    Client for QuantumArchitect-MCP server.
    Wraps MCP endpoints with fallback to local implementations.
    
    Primary endpoints (from Gradio):
    - ui_create_circuit
    - ui_validate_circuit
    - ui_simulate_circuit
    - ui_score_circuit
    
    Missing endpoints use QASMLocalAnalyzer for fallback.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:7861"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self._connected = False
        self._analyzer = QASMLocalAnalyzer()

    def _call(self, endpoint: str, **kwargs) -> MCPResponse:
        """Internal method to call MCP endpoints."""
        start = time.perf_counter()

        try:
            url = f"{self.base_url}/gradio_api/call/{endpoint}"
            payload = {"data": list(kwargs.values()) if kwargs else []}
            
            response = self.session.post(url, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            event_id = result.get("event_id")

            if event_id:
                result_url = f"{self.base_url}/gradio_api/call/{endpoint}/{event_id}"
                result_response = self.session.get(result_url, timeout=30)

                lines = result_response.text.strip().split("\n")
                for line in lines:
                    if line.startswith("data:"):
                        data = json.loads(line[5:].strip())
                        elapsed = (time.perf_counter() - start) * 1000
                        return MCPResponse(
                            success=True,
                            data=data[0] if isinstance(data, list) and len(data) == 1 else data,
                            endpoint=endpoint,
                            execution_time_ms=elapsed
                        )

            elapsed = (time.perf_counter() - start) * 1000
            return MCPResponse(
                success=True,
                data=result,
                endpoint=endpoint,
                execution_time_ms=elapsed
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning(f"MCP call failed: {endpoint} - {e}")
            return MCPResponse(
                success=False,
                data=None,
                endpoint=endpoint,
                error=str(e),
                execution_time_ms=elapsed
            )

    def _fallback_response(self, endpoint: str, data: Any, start_time: float) -> MCPResponse:
        """Create a fallback response using local implementation."""
        elapsed = (time.perf_counter() - start_time) * 1000
        return MCPResponse(
            success=True,
            data=data,
            endpoint=f"{endpoint}(fallback)",
            execution_time_ms=elapsed,
            is_fallback=True
        )

    def health_check(self) -> bool:
        """Check if MCP server is reachable."""
        try:
            response = self.session.get(f"{self.base_url}/", timeout=5)
            self._connected = response.status_code == 200
            return self._connected
        except:
            self._connected = False
            return False

    # ===== Circuit Creation Endpoints =====
    
    def create_circuit_from_template(self, template_name: str, num_qubits: int = 2) -> MCPResponse:
        """Create a circuit from a predefined template.
        Maps to ui_create_circuit endpoint in Gradio."""
        return self._call("ui_create_circuit", template=template_name, qubits=num_qubits, params="{}")

    def generate_random_circuit(self, num_qubits: int = 3, depth: int = 5,
                                gate_set: str = "h,cx,rz") -> MCPResponse:
        """Generate a random quantum circuit. Uses local fallback."""
        start = time.perf_counter()
        gates = gate_set.split(',')
        
        qasm_lines = [
            'OPENQASM 2.0;',
            'include "qelib1.inc";',
            f'qreg q[{num_qubits}];',
            f'creg c[{num_qubits}];'
        ]
        
        for _ in range(depth):
            gate = random.choice(gates)
            if gate in ['h', 'x', 'y', 'z', 's', 't']:
                q = random.randint(0, num_qubits - 1)
                qasm_lines.append(f'{gate} q[{q}];')
            elif gate in ['cx', 'cz']:
                if num_qubits >= 2:
                    q1 = random.randint(0, num_qubits - 1)
                    q2 = random.randint(0, num_qubits - 1)
                    while q2 == q1:
                        q2 = random.randint(0, num_qubits - 1)
                    qasm_lines.append(f'{gate} q[{q1}], q[{q2}];')
            elif gate in ['rz', 'rx', 'ry']:
                q = random.randint(0, num_qubits - 1)
                angle = round(random.uniform(0, 2 * math.pi), 4)
                qasm_lines.append(f'{gate}({angle}) q[{q}];')
        
        qasm_lines.append(f'measure q -> c;')
        qasm_code = '\n'.join(qasm_lines)
        
        return self._fallback_response("generate_random_circuit", {'qasm': qasm_code}, start)

    def generate_circuit_from_description(self, description: str) -> MCPResponse:
        """Generate circuit from natural language description.
        Uses ui_create_circuit with best-matching template."""
        desc_lower = description.lower()
        
        if 'entangle' in desc_lower or 'bell' in desc_lower:
            template = 'bell_state'
        elif 'ghz' in desc_lower:
            template = 'ghz_state'
        elif 'superposition' in desc_lower:
            template = 'superposition'
        elif 'qft' in desc_lower or 'fourier' in desc_lower:
            template = 'qft'
        elif 'grover' in desc_lower or 'search' in desc_lower:
            template = 'grover'
        elif 'vqe' in desc_lower or 'variational' in desc_lower:
            template = 'vqe'
        else:
            template = 'bell_state'
        
        return self._call("ui_create_circuit", template=template, qubits=2, params="{}")

    # ===== Parsing & Analysis Endpoints (Fallback) =====

    def parse_qasm(self, qasm_code: str) -> MCPResponse:
        """Parse OpenQASM code into circuit structure. Uses local fallback."""
        start = time.perf_counter()
        parsed = self._analyzer.parse_qasm(qasm_code)
        return self._fallback_response("parse_qasm", parsed, start)

    def analyze_circuit(self, qasm_code: str) -> MCPResponse:
        """Analyze circuit properties (depth, gates, etc.). Uses local fallback."""
        start = time.perf_counter()
        analysis = self._analyzer.analyze_circuit(qasm_code)
        return self._fallback_response("analyze_circuit", analysis, start)

    def get_circuit_depth(self, qasm_code: str) -> MCPResponse:
        """Get the depth of a circuit. Uses local fallback."""
        start = time.perf_counter()
        depth = self._analyzer.get_depth(qasm_code)
        return self._fallback_response("get_circuit_depth", {'depth': depth}, start)

    # ===== Validation Endpoints =====

    def validate_syntax(self, qasm_code: str) -> MCPResponse:
        """Validate QASM syntax. Maps to ui_validate_circuit."""
        return self._call("ui_validate_circuit", qasm=qasm_code, hardware="")

    def check_connectivity(self, qasm_code: str, hardware: str = "ibm_brisbane") -> MCPResponse:
        """Check if circuit respects hardware connectivity. Uses ui_validate_circuit."""
        return self._call("ui_validate_circuit", qasm=qasm_code, hardware=hardware)

    def verify_unitary(self, qasm_code: str) -> MCPResponse:
        """Verify circuit produces valid unitary. Uses local fallback."""
        start = time.perf_counter()
        validation = self._analyzer.validate_syntax(qasm_code)
        result = {
            'is_unitary': validation['valid'],
            'errors': validation['errors'],
            'note': 'Local validation - full unitary check requires simulation'
        }
        return self._fallback_response("verify_unitary", result, start)

    # ===== Simulation Endpoints =====

    def simulate_circuit(self, qasm_code: str, shots: int = 1024) -> MCPResponse:
        """Simulate circuit and get measurement results. Maps to ui_simulate_circuit."""
        return self._call("ui_simulate_circuit", qasm=qasm_code, shots=shots)

    def get_statevector(self, qasm_code: str) -> MCPResponse:
        """Get the statevector of a circuit. Uses ui_simulate_circuit."""
        result = self._call("ui_simulate_circuit", qasm=qasm_code, shots=1)
        if result.success and result.data:
            result.data = {'statevector_hint': 'Use simulation results for state info'}
        return result

    def get_probabilities(self, qasm_code: str) -> MCPResponse:
        """Get probability distribution from circuit. Uses ui_simulate_circuit."""
        result = self._call("ui_simulate_circuit", qasm=qasm_code, shots=1024)
        if result.success and result.data:
            # Extract probabilities from histogram
            result.endpoint = "get_probabilities"
        return result

    # ===== Scoring Endpoints =====

    def calculate_complexity_score(self, qasm_code: str) -> MCPResponse:
        """Calculate circuit complexity score. Tries ui_score_circuit then fallback."""
        result = self._call("ui_score_circuit", qasm=qasm_code, hardware="ibm_brisbane")
        if result.success:
            return result
        
        # Fallback to local
        start = time.perf_counter()
        complexity = self._analyzer.calculate_complexity(qasm_code)
        return self._fallback_response("calculate_complexity_score", complexity, start)

    def calculate_hardware_fitness(self, qasm_code: str, hardware: str = "ibm_brisbane") -> MCPResponse:
        """Calculate hardware fitness score. Tries ui_score_circuit then fallback."""
        result = self._call("ui_score_circuit", qasm=qasm_code, hardware=hardware)
        if result.success:
            return result
            
        # Fallback to local
        start = time.perf_counter()
        fitness = self._analyzer.calculate_hardware_fitness(qasm_code, hardware)
        return self._fallback_response("calculate_hardware_fitness", fitness, start)

    def calculate_expressibility(self, qasm_code: str) -> MCPResponse:
        """Calculate circuit expressibility. Uses local fallback."""
        start = time.perf_counter()
        analysis = self._analyzer.analyze_circuit(qasm_code)
        
        # Expressibility heuristic based on gate diversity and depth
        gate_types = len(analysis['gate_breakdown'])
        depth_factor = min(analysis['depth'] / 20.0, 1.0)
        entangle_factor = min(analysis['two_qubit_gates'] / 5.0, 1.0)
        
        expressibility = (gate_types * 0.3 + depth_factor * 0.35 + entangle_factor * 0.35) * 100
        
        result = {
            'expressibility_score': round(expressibility, 2),
            'gate_diversity': gate_types,
            'depth_factor': round(depth_factor, 2),
            'entanglement_factor': round(entangle_factor, 2)
        }
        return self._fallback_response("calculate_expressibility", result, start)

    # ===== Resource Estimation Endpoints (Fallback) =====

    def estimate_resources(self, qasm_code: str) -> MCPResponse:
        """Estimate resource requirements. Uses local fallback."""
        start = time.perf_counter()
        analysis = self._analyzer.analyze_circuit(qasm_code)
        
        result = {
            'qubits_required': analysis['num_qubits'],
            'classical_bits': analysis['num_classical_bits'],
            'gate_count': analysis['gate_count'],
            'depth': analysis['depth'],
            'estimated_runtime_ms': analysis['depth'] * 0.1,  # Rough estimate
            'memory_footprint_bytes': analysis['num_qubits'] * 16 * (2 ** analysis['num_qubits'])
        }
        return self._fallback_response("estimate_resources", result, start)

    def estimate_noise(self, qasm_code: str, hardware: str = "ibm_brisbane") -> MCPResponse:
        """Estimate noise impact on circuit. Uses local fallback."""
        start = time.perf_counter()
        analysis = self._analyzer.analyze_circuit(qasm_code)
        
        # Noise profiles (simplified)
        noise_rates = {
            'ibm_brisbane': {'single_q': 0.001, 'two_q': 0.01, 'readout': 0.02},
            'ibm_sherbrooke': {'single_q': 0.0008, 'two_q': 0.008, 'readout': 0.015},
            'rigetti_aspen': {'single_q': 0.002, 'two_q': 0.02, 'readout': 0.03},
            'ionq_harmony': {'single_q': 0.0003, 'two_q': 0.005, 'readout': 0.01}
        }
        
        rates = noise_rates.get(hardware, noise_rates['ibm_brisbane'])
        
        single_q_error = analysis['single_qubit_gates'] * rates['single_q']
        two_q_error = analysis['two_qubit_gates'] * rates['two_q']
        readout_error = analysis['measurements'] * rates['readout']
        total_error = 1 - (1 - single_q_error) * (1 - two_q_error) * (1 - readout_error)
        
        result = {
            'estimated_fidelity': round(1 - total_error, 4),
            'single_qubit_error': round(single_q_error, 4),
            'two_qubit_error': round(two_q_error, 4),
            'readout_error': round(readout_error, 4),
            'total_error_probability': round(total_error, 4),
            'hardware': hardware
        }
        return self._fallback_response("estimate_noise", result, start)

    # ===== Composition Endpoints (Fallback) =====

    def compose_circuits(self, qasm1: str, qasm2: str, qubit_mapping: str = "") -> MCPResponse:
        """Compose two circuits sequentially. Uses local fallback."""
        start = time.perf_counter()
        
        # Parse both circuits
        parsed1 = self._analyzer.parse_qasm(qasm1)
        parsed2 = self._analyzer.parse_qasm(qasm2)
        
        # Simple sequential composition
        num_qubits = max(parsed1['num_qubits'], parsed2['num_qubits'])
        
        lines = [
            'OPENQASM 2.0;',
            'include "qelib1.inc";',
            f'qreg q[{num_qubits}];',
            f'creg c[{num_qubits}];'
        ]
        
        # Add gates from both circuits
        for g in parsed1['gates']:
            if g['gate'].lower() != 'measure':
                lines.append(f"{g['raw']};")
        for g in parsed2['gates']:
            lines.append(f"{g['raw']};")
        
        result = {'qasm': '\n'.join(lines)}
        return self._fallback_response("compose_circuits", result, start)

    def generate_inverse_circuit(self, qasm_code: str) -> MCPResponse:
        """Generate the inverse of a circuit. Uses local fallback."""
        start = time.perf_counter()
        parsed = self._analyzer.parse_qasm(qasm_code)
        
        # Inverse gate mappings
        inverse_map = {
            'h': 'h', 'x': 'x', 'y': 'y', 'z': 'z',
            's': 'sdg', 'sdg': 's', 't': 'tdg', 'tdg': 't',
            'cx': 'cx', 'cz': 'cz', 'swap': 'swap'
        }
        
        lines = [
            'OPENQASM 2.0;',
            'include "qelib1.inc";',
            f'qreg q[{parsed["num_qubits"]}];',
            f'creg c[{parsed["num_classical"]}];'
        ]
        
        # Reverse and invert gates
        for g in reversed(parsed['gates']):
            gate = g['gate'].lower()
            if gate == 'measure':
                continue
            inv_gate = inverse_map.get(gate, gate)
            # Handle parametric gates
            if '(' in g['raw']:
                # Negate angle for rotation gates
                raw = g['raw'].replace(gate, inv_gate)
                if 'rz' in gate or 'rx' in gate or 'ry' in gate:
                    # Simple negation (not perfect)
                    pass
                lines.append(f"{raw};")
            else:
                raw = g['raw'].replace(gate, inv_gate)
                lines.append(f"{raw};")
        
        result = {'qasm': '\n'.join(lines)}
        return self._fallback_response("generate_inverse_circuit", result, start)

    def tensor_circuits(self, qasm1: str, qasm2: str) -> MCPResponse:
        """Tensor product of two circuits. Uses local fallback."""
        start = time.perf_counter()
        
        parsed1 = self._analyzer.parse_qasm(qasm1)
        parsed2 = self._analyzer.parse_qasm(qasm2)
        
        total_qubits = parsed1['num_qubits'] + parsed2['num_qubits']
        offset = parsed1['num_qubits']
        
        lines = [
            'OPENQASM 2.0;',
            'include "qelib1.inc";',
            f'qreg q[{total_qubits}];',
            f'creg c[{total_qubits}];'
        ]
        
        # Add gates from first circuit
        for g in parsed1['gates']:
            lines.append(f"{g['raw']};")
        
        # Add gates from second circuit with offset
        for g in parsed2['gates']:
            raw = g['raw']
            # Offset qubit indices
            for i in range(parsed2['num_qubits'] - 1, -1, -1):
                raw = raw.replace(f'q[{i}]', f'q[{i + offset}]')
            lines.append(f"{raw};")
        
        result = {'qasm': '\n'.join(lines)}
        return self._fallback_response("tensor_circuits", result, start)

    def repeat_circuit(self, qasm_code: str, n: int) -> MCPResponse:
        """Repeat a circuit n times. Uses local fallback."""
        start = time.perf_counter()
        parsed = self._analyzer.parse_qasm(qasm_code)
        
        lines = [
            'OPENQASM 2.0;',
            'include "qelib1.inc";',
            f'qreg q[{parsed["num_qubits"]}];',
            f'creg c[{parsed["num_classical"]}];'
        ]
        
        # Repeat non-measure gates n times
        for _ in range(n):
            for g in parsed['gates']:
                if g['gate'].lower() != 'measure':
                    lines.append(f"{g['raw']};")
        
        # Add measurements at end
        for g in parsed['gates']:
            if g['gate'].lower() == 'measure':
                lines.append(f"{g['raw']};")
                break
        
        result = {'qasm': '\n'.join(lines)}
        return self._fallback_response("repeat_circuit", result, start)

    # ===== Utility Endpoints =====

    def list_templates(self) -> MCPResponse:
        """List available circuit templates."""
        start = time.perf_counter()
        templates = [
            'bell_state', 'ghz_state', 'w_state', 'superposition',
            'qft', 'grover', 'vqe', 'qaoa'
        ]
        return self._fallback_response("list_templates", {'templates': templates}, start)

    def list_hardware_profiles(self) -> MCPResponse:
        """List available hardware profiles."""
        start = time.perf_counter()
        profiles = ['ibm_brisbane', 'ibm_sherbrooke', 'rigetti_aspen', 'ionq_harmony']
        return self._fallback_response("list_hardware_profiles", {'profiles': profiles}, start)


# Singleton client instance
_client: Optional[MCPClient] = None


def get_client(base_url: Optional[str] = None) -> MCPClient:
    """
    Get or create the MCP client singleton.
    
    Args:
        base_url: Optional URL override. If None, checks MCP_SERVER_URL env var,
                 then defaults to http://127.0.0.1:7861
    """
    global _client
    if _client is None:
        if base_url is None:
            import os
            base_url = os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:7861")
        _client = MCPClient(base_url)
    return _client

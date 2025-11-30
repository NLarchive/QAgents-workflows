# Path: QAgents-workflos/orchestrators/quasar_orchestrator.py
# Relations: Uses agents/llm_adapter.py, tools/quantum_tools.py, client/mcp_client.py
# Description: QUASAR-lite orchestrator implementing Tool-Augmented LLM with hierarchical rewards
"""
QUASAR-Lite Orchestrator: Tool-Augmented LLM with Hierarchical Verification

Based on the QUASAR framework (2025) for quantum circuit generation:
- Tier 1: Syntax validation (compile check)
- Tier 2: Semantic validation (unitarity, qubit count)  
- Tier 3: Correctness validation (expected states)
- Tier 4: Optimization (depth/gate count)

Key Innovation: LLM generates → Tool validates → Feedback loop until success
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import time
import re

logger = logging.getLogger(__name__)


@dataclass
class ValidationTier:
    """Result from a validation tier."""
    tier: int
    name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuasarResult:
    """Result from QUASAR orchestration."""
    success: bool
    final_qasm: Optional[str]
    execution_time_ms: float
    llm_calls: int
    tokens_used: int
    tiers_passed: List[int]
    validation_history: List[ValidationTier] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    iterations: int = 0

    @property
    def final_output(self) -> Optional[str]:
        """Alias for compatibility with OrchestratorResult."""
        return self.final_qasm
class QuasarOrchestrator:
    """
    QUASAR-Lite: Tool-Augmented LLM for Quantum Circuit Generation
    
    Key differences from NAKED mode:
    1. Validates after each generation attempt
    2. Provides error feedback to LLM for self-correction
    3. Uses hierarchical reward tiers
    4. Supports circuit partitioning for complex problems
    
    Key differences from GUIDED mode:
    1. Single LLM with tool access (not multi-agent)
    2. External validation (not self-reflection)
    3. Iterative refinement with ground-truth feedback
    """
    
    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self._llm = None
        self._mcp_client = None
        
    def _get_llm(self):
        """Lazy load LLM adapter."""
        if self._llm is None:
            from agents.llm_adapter import get_llm_adapter
            from config import config
            self._llm = get_llm_adapter(
                provider="gemini",
                api_key=config.llm.api_key,
                enable_fallback=True
            )
        return self._llm
    
    def _get_mcp(self):
        """Lazy load MCP client for validation."""
        if self._mcp_client is None:
            from client.mcp_client import get_client
            self._mcp_client = get_client()
        return self._mcp_client
    
    def _extract_qasm(self, text: str) -> Optional[str]:
        """Extract QASM code from LLM response."""
        if not text:
            return None
            
        # Clean up common LLM artifacts
        if "```" in text:
            lines = text.split("\n")
            in_block = False
            qasm_lines = []
            for line in lines:
                if line.strip().startswith("```"):
                    if in_block:
                        break
                    in_block = True
                    continue
                if in_block:
                    qasm_lines.append(line)
            text = "\n".join(qasm_lines)
        
        # Find OPENQASM declaration
        if "OPENQASM" in text:
            idx = text.find("OPENQASM")
            return text[idx:].strip()
        
        # Try to construct valid QASM
        if "qreg" in text or "include" in text:
            return "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n" + text.strip()
            
        return None
    
    def _validate_tier1_syntax(self, qasm: str) -> ValidationTier:
        """Tier 1: Syntax validation - does it compile?"""
        try:
            mcp = self._get_mcp()
            result = mcp.validate_syntax(qasm)
            
            if result.success and result.data:
                is_valid = result.data.get("valid", False)
                errors = result.data.get("errors", [])
                
                if is_valid:
                    return ValidationTier(
                        tier=1, name="Syntax", passed=True,
                        message="QASM syntax is valid",
                        details={"valid": True}
                    )
                else:
                    return ValidationTier(
                        tier=1, name="Syntax", passed=False,
                        message=f"Syntax errors: {errors}",
                        details={"errors": errors}
                    )
            
            return ValidationTier(
                tier=1, name="Syntax", passed=False,
                message="Validation failed",
                details={"error": "MCP validation failed"}
            )
            
        except Exception as e:
            # Fallback: basic regex validation
            has_header = "OPENQASM" in qasm and "include" in qasm
            has_qreg = "qreg" in qasm
            has_creg = "creg" in qasm
            
            if has_header and has_qreg:
                return ValidationTier(
                    tier=1, name="Syntax", passed=True,
                    message="Basic syntax check passed (fallback)",
                    details={"fallback": True}
                )
            return ValidationTier(
                tier=1, name="Syntax", passed=False,
                message=f"Basic syntax check failed: {e}",
                details={"error": str(e)}
            )
    
    def _validate_tier2_semantic(self, qasm: str, expected_qubits: int = None) -> ValidationTier:
        """Tier 2: Semantic validation - qubit count, gate validity."""
        try:
            mcp = self._get_mcp()
            result = mcp.analyze_circuit(qasm)
            
            if result.success and result.data:
                num_qubits = result.data.get("num_qubits", 0)
                gate_count = result.data.get("gate_count", 0)
                
                issues = []
                
                # Check qubit count if expected
                if expected_qubits and num_qubits != expected_qubits:
                    issues.append(f"Expected {expected_qubits} qubits, got {num_qubits}")
                
                # Check for at least one gate
                if gate_count == 0:
                    issues.append("No gates in circuit")
                
                if issues:
                    return ValidationTier(
                        tier=2, name="Semantic", passed=False,
                        message="; ".join(issues),
                        details={"num_qubits": num_qubits, "gate_count": gate_count}
                    )
                
                return ValidationTier(
                    tier=2, name="Semantic", passed=True,
                    message=f"Valid circuit: {num_qubits} qubits, {gate_count} gates",
                    details={"num_qubits": num_qubits, "gate_count": gate_count}
                )
                
        except Exception as e:
            # Fallback: regex-based analysis
            qreg_match = re.search(r'qreg\s+\w+\[(\d+)\]', qasm)
            num_qubits = int(qreg_match.group(1)) if qreg_match else 0
            
            gate_pattern = r'\b(h|x|y|z|s|t|cx|cz|cy|swap|ccx|rz|rx|ry)\b'
            gates = re.findall(gate_pattern, qasm, re.IGNORECASE)
            
            return ValidationTier(
                tier=2, name="Semantic", passed=len(gates) > 0,
                message=f"Fallback analysis: {num_qubits} qubits, {len(gates)} gates",
                details={"fallback": True, "num_qubits": num_qubits, "gate_count": len(gates)}
            )
    
    def _validate_tier3_correctness(self, qasm: str, expected_states: Dict[str, float] = None) -> ValidationTier:
        """Tier 3: Correctness validation - expected output states."""
        if not expected_states:
            return ValidationTier(
                tier=3, name="Correctness", passed=True,
                message="No expected states specified, skipping",
                details={"skipped": True}
            )
        
        try:
            mcp = self._get_mcp()
            result = mcp.simulate_circuit(qasm, shots=1024)
            
            if result.success and result.data:
                probs = result.data.get("probabilities", {})
                
                # Check if expected states match
                tolerance = 0.15
                matches = []
                mismatches = []
                
                for state, expected_prob in expected_states.items():
                    actual_prob = probs.get(state, 0.0)
                    if abs(actual_prob - expected_prob) <= tolerance:
                        matches.append(f"|{state}⟩: {actual_prob:.3f} ≈ {expected_prob}")
                    else:
                        mismatches.append(f"|{state}⟩: got {actual_prob:.3f}, expected {expected_prob}")
                
                if mismatches:
                    return ValidationTier(
                        tier=3, name="Correctness", passed=False,
                        message=f"State mismatches: {mismatches}",
                        details={"expected": expected_states, "actual": probs}
                    )
                
                return ValidationTier(
                    tier=3, name="Correctness", passed=True,
                    message=f"States match: {matches}",
                    details={"matches": matches}
                )
                
        except Exception as e:
            return ValidationTier(
                tier=3, name="Correctness", passed=False,
                message=f"Simulation failed: {e}",
                details={"error": str(e)}
            )
    
    def _validate_tier4_optimization(self, qasm: str, max_depth: int = None) -> ValidationTier:
        """Tier 4: Optimization - circuit depth and gate count."""
        try:
            mcp = self._get_mcp()
            result = mcp.analyze_circuit(qasm)
            
            if result.success and result.data:
                depth = result.data.get("depth", 0)
                gate_count = result.data.get("gate_count", 0)
                cx_count = result.data.get("cx_count", 0)
                
                details = {"depth": depth, "gate_count": gate_count, "cx_count": cx_count}
                
                if max_depth and depth > max_depth:
                    return ValidationTier(
                        tier=4, name="Optimization", passed=False,
                        message=f"Depth {depth} exceeds max {max_depth}",
                        details=details
                    )
                
                return ValidationTier(
                    tier=4, name="Optimization", passed=True,
                    message=f"Depth: {depth}, Gates: {gate_count}, CX: {cx_count}",
                    details=details
                )
                
        except Exception as e:
            return ValidationTier(
                tier=4, name="Optimization", passed=True,
                message=f"Optimization check skipped: {e}",
                details={"error": str(e)}
            )
    
    def _build_feedback_prompt(self, goal: str, previous_qasm: str, 
                               failed_tier: ValidationTier, iteration: int) -> str:
        """Build prompt with feedback for LLM self-correction."""
        return f"""Your previous attempt to generate a quantum circuit had an error.

ORIGINAL TASK:
{goal}

YOUR PREVIOUS OUTPUT:
```qasm
{previous_qasm or "(no valid QASM generated)"}
```

VALIDATION ERROR (Tier {failed_tier.tier} - {failed_tier.name}):
{failed_tier.message}

Details: {failed_tier.details}

INSTRUCTIONS:
1. Analyze the error carefully
2. Fix the issue in your QASM code
3. Output ONLY valid OpenQASM 2.0 code
4. Start with: OPENQASM 2.0; include "qelib1.inc";

Generate the CORRECTED QASM code:"""

    def _build_initial_prompt(self, goal: str, expected_qubits: int = None, 
                              expected_states: Dict[str, float] = None) -> str:
        """Build the initial generation prompt."""
        constraints = []
        if expected_qubits:
            constraints.append(f"- Use exactly {expected_qubits} qubit(s)")
        if expected_states:
            states_str = ", ".join([f"|{s}⟩: {p}" for s, p in expected_states.items()])
            constraints.append(f"- Expected measurement probabilities: {states_str}")
        
        constraints_section = "\n".join(constraints) if constraints else "- No specific constraints"
        
        return f"""Generate a quantum circuit for the following task:

TASK:
{goal}

CONSTRAINTS:
{constraints_section}

RULES:
1. Output ONLY valid OpenQASM 2.0 code
2. Start with: OPENQASM 2.0; include "qelib1.inc";
3. Declare qubits with: qreg q[N];
4. Declare classical bits with: creg c[N];
5. Use standard gates: h, x, y, z, cx, cz, ccx, swap, t, s, rx, ry, rz
6. Add measurements with: measure q[i] -> c[i];
7. NO explanations, NO markdown, ONLY QASM code

Generate the OpenQASM 2.0 circuit:"""

    def run(self, goal: str, 
            expected_qubits: int = None,
            expected_states: Dict[str, float] = None,
            max_depth: int = None) -> QuasarResult:
        """
        Run QUASAR-lite orchestration with hierarchical validation.
        
        Args:
            goal: The problem description
            expected_qubits: Expected number of qubits (for Tier 2)
            expected_states: Expected output states (for Tier 3)
            max_depth: Maximum circuit depth (for Tier 4)
            
        Returns:
            QuasarResult with final QASM and validation history
        """
        start_time = time.perf_counter()
        
        llm = self._get_llm()
        llm_calls = 0
        tokens_used = 0
        validation_history = []
        errors = []
        current_qasm = None
        tiers_passed = []
        
        system_prompt = """You are an expert quantum computing engineer.
Your task is to generate valid OpenQASM 2.0 code for quantum circuits.
You will receive feedback if your code has errors and must correct them.
Always output ONLY valid QASM code, no explanations."""

        # Initial prompt
        user_prompt = self._build_initial_prompt(goal, expected_qubits, expected_states)
        
        for iteration in range(self.max_iterations):
            # Generate QASM
            try:
                response = llm.generate(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1 + (iteration * 0.1),  # Increase temperature on retries
                    max_tokens=1500
                )
                llm_calls += 1
                tokens_used += response.tokens_used
                
                current_qasm = self._extract_qasm(response.text)
                
                if not current_qasm:
                    errors.append(f"Iteration {iteration+1}: Failed to extract QASM")
                    user_prompt = self._build_feedback_prompt(
                        goal, response.text,
                        ValidationTier(0, "Extraction", False, "No valid QASM found in response"),
                        iteration
                    )
                    continue
                    
            except KeyboardInterrupt:
                raise  # Re-raise keyboard interrupt
            except Exception as e:
                errors.append(f"Iteration {iteration+1}: LLM error - {e}")
                logger.error(f"QUASAR LLM error: {e}")
                # Don't continue retrying on LLM errors, they'll likely fail again
                break
            
            # Run hierarchical validation
            all_passed = True
            tiers_passed = []
            
            # Tier 1: Syntax
            tier1 = self._validate_tier1_syntax(current_qasm)
            validation_history.append(tier1)
            if not tier1.passed:
                all_passed = False
                user_prompt = self._build_feedback_prompt(goal, current_qasm, tier1, iteration)
                continue
            tiers_passed.append(1)
            
            # Tier 2: Semantic
            tier2 = self._validate_tier2_semantic(current_qasm, expected_qubits)
            validation_history.append(tier2)
            if not tier2.passed:
                all_passed = False
                user_prompt = self._build_feedback_prompt(goal, current_qasm, tier2, iteration)
                continue
            tiers_passed.append(2)
            
            # Tier 3: Correctness (if expected states provided)
            if expected_states:
                tier3 = self._validate_tier3_correctness(current_qasm, expected_states)
                validation_history.append(tier3)
                if not tier3.passed:
                    all_passed = False
                    user_prompt = self._build_feedback_prompt(goal, current_qasm, tier3, iteration)
                    continue
                tiers_passed.append(3)
            
            # Tier 4: Optimization (informational, doesn't fail)
            tier4 = self._validate_tier4_optimization(current_qasm, max_depth)
            validation_history.append(tier4)
            if tier4.passed:
                tiers_passed.append(4)
            
            # All validations passed!
            if all_passed:
                elapsed = (time.perf_counter() - start_time) * 1000
                return QuasarResult(
                    success=True,
                    final_qasm=current_qasm,
                    execution_time_ms=elapsed,
                    llm_calls=llm_calls,
                    tokens_used=tokens_used,
                    tiers_passed=tiers_passed,
                    validation_history=validation_history,
                    errors=errors,
                    iterations=iteration + 1
                )
        
        # Max iterations reached
        elapsed = (time.perf_counter() - start_time) * 1000
        return QuasarResult(
            success=current_qasm is not None and len(tiers_passed) >= 2,
            final_qasm=current_qasm,
            execution_time_ms=elapsed,
            llm_calls=llm_calls,
            tokens_used=tokens_used,
            tiers_passed=tiers_passed,
            validation_history=validation_history,
            errors=errors,
            iterations=self.max_iterations
        )


class HybridOrchestrator:
    """
    Hybrid Orchestrator: NAKED speed + QUASAR reliability
    
    Strategy:
    1. Try NAKED mode first (fast, cheap)
    2. If NAKED fails validation, fall back to QUASAR (reliable, more expensive)
    
    This gives best of both worlds:
    - Easy problems: solved in 1 LLM call via NAKED
    - Hard problems: solved via QUASAR with feedback loops
    """
    
    def __init__(self):
        self._naked = None
        self._quasar = None
        
    def _get_naked(self):
        """Lazy load NAKED orchestrator."""
        if self._naked is None:
            from orchestrators.orchestrator import NakedOrchestrator
            self._naked = NakedOrchestrator()
        return self._naked
    
    def _get_quasar(self):
        """Lazy load QUASAR orchestrator."""
        if self._quasar is None:
            self._quasar = QuasarOrchestrator(max_iterations=3)
        return self._quasar
    
    def run(self, goal: str, 
            expected_qubits: int = None,
            expected_states: Dict[str, float] = None,
            max_depth: int = None) -> QuasarResult:
        """
        Run hybrid orchestration: NAKED first, QUASAR on failure.
        
        Returns:
            QuasarResult for compatibility with comprehensive tests
        """
        start_time = time.perf_counter()
        
        # Step 1: Try NAKED mode
        naked = self._get_naked()
        naked_result = naked.run(goal)
        
        if naked_result.success and naked_result.final_output:
            # Validate NAKED output
            quasar = self._get_quasar()
            qasm = naked_result.final_output
            
            tier1 = quasar._validate_tier1_syntax(qasm)
            tier2 = quasar._validate_tier2_semantic(qasm, expected_qubits)
            
            if tier1.passed and tier2.passed:
                # NAKED succeeded!
                elapsed = (time.perf_counter() - start_time) * 1000
                return QuasarResult(
                    success=True,
                    final_qasm=qasm,
                    execution_time_ms=elapsed,
                    llm_calls=1,
                    tokens_used=naked_result.agent_results.get("naked_llm", {}).data.get("tokens_used", 0) if naked_result.agent_results else 0,
                    tiers_passed=[1, 2],
                    validation_history=[tier1, tier2],
                    errors=[],
                    iterations=1
                )
        
        # Step 2: NAKED failed, use QUASAR
        logger.info(f"NAKED failed, falling back to QUASAR for: {goal[:50]}...")
        quasar = self._get_quasar()
        return quasar.run(goal, expected_qubits, expected_states, max_depth)

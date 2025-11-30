"""
QAgents-Workflows: Configuration
Central configuration for the multi-agent quantum circuit optimization system.

Path: QAgents-workflos/config.py
Related: agents/llm_adapter.py (uses GEMINI_MODELS for fallback cascade)
         run_evaluation.py (uses config for evaluation settings)
         workflows/workflow_definitions.py (references rate limits)
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import os

# Paths
PROJECT_ROOT = Path(__file__).parent
QUANTUM_MCP_ROOT = PROJECT_ROOT.parent / "QuantumArchitect-MCP"

# =============================================================================
# GEMINI MODEL CASCADE (sorted by RPD - highest to lowest for optimal fallback)
# =============================================================================
# When a model hits rate limits (RPM/RPD), fallback to next model in list.
# Free tier limits (as of 2025):
#   - Gemma 3:             30 RPM, 15K TPM, 14,400 RPD (HIGHEST availability)
#   - Flash-Lite:          15 RPM, 250K TPM, 1,000 RPD
#   - Flash 2.5:           10 RPM, 250K TPM, 250 RPD
#   - Flash 2.0:           15 RPM, 1M TPM, 200 RPD
#   - Flash 2.0 Lite:      30 RPM, 1M TPM, 200 RPD
#   - Pro 2.5:             2 RPM, 125K TPM, 50 RPD (LOWEST availability)
#
# EXPECTED REQUESTS PER EVALUATION (9 problems):
#   - Naked mode:     0 LLM calls (direct MCP only)
#   - Guided mode:    ~36 LLM calls (4 per problem)
#   - Blackboard:     ~72-108 LLM calls (8-12 per problem)
# =============================================================================

GEMINI_MODELS: List[Dict] = [
    # Highest RPD - most available (14,400/day = 10/min continuously)
    {
        "name": "gemma-3-27b-it",
        "rpm": 30,
        "tpm": 15_000,
        "rpd": 14_400,
        "priority": 1,
        "notes": "Best for high-volume, may have lower quality than Flash"
    },
    # Good balance - default model (1,000/day)
    {
        "name": "gemini-2.5-flash-lite",
        "rpm": 15,
        "tpm": 250_000,
        "rpd": 1_000,
        "priority": 2,
        "notes": "Good balance of quality and availability - DEFAULT"
    },
    # Higher quality - moderate availability (250/day)
    {
        "name": "gemini-2.5-flash",
        "rpm": 10,
        "tpm": 250_000,
        "rpd": 250,
        "priority": 3,
        "notes": "Better quality, lower availability"
    },
    # High TPM for long contexts (200/day)
    {
        "name": "gemini-2.0-flash",
        "rpm": 15,
        "tpm": 1_000_000,
        "rpd": 200,
        "priority": 4,
        "notes": "Good for long contexts, moderate availability"
    },
    # Fast variant (200/day)
    {
        "name": "gemini-2.0-flash-lite",
        "rpm": 30,
        "tpm": 1_000_000,
        "rpd": 200,
        "priority": 5,
        "notes": "Fast responses, lower availability"
    },
    # Lowest RPD - highest quality, use sparingly (50/day)
    {
        "name": "gemini-2.5-pro",
        "rpm": 2,
        "tpm": 125_000,
        "rpd": 50,
        "priority": 6,
        "notes": "Highest quality, use sparingly - LAST RESORT"
    },
]

def get_model_by_priority(priority: int = 1) -> Optional[Dict]:
    """Get model config by priority (1=highest RPD)."""
    for model in GEMINI_MODELS:
        if model["priority"] == priority:
            return model
    return None

def get_next_model(current_name: str) -> Optional[Dict]:
    """Get next model in fallback chain."""
    for i, model in enumerate(GEMINI_MODELS):
        if model["name"] == current_name:
            if i + 1 < len(GEMINI_MODELS):
                return GEMINI_MODELS[i + 1]
    return None

def get_model_config(model_name: str) -> Optional[Dict]:
    """Get model config by name."""
    for model in GEMINI_MODELS:
        if model["name"] == model_name:
            return model
    return None


@dataclass
class MCPConfig:
    """MCP Server configuration."""
    host: str = "127.0.0.1"
    port: int = 7861
    base_url: str = field(init=False)

    def __post_init__(self):
        self.base_url = f"http://{self.host}:{self.port}"


@dataclass
class RateLimitConfig:
    """Rate limiting based on Gemini API free tier limits."""
    # Default to gemini-2.5-flash-lite limits
    rpm_limit: int = 15  # Requests per minute
    tpm_limit: int = 250_000  # Tokens per minute
    rpd_limit: int = 1_000  # Requests per day

    # Conservative buffer (80% of limit = 12 RPM effective)
    rpm_buffer: float = 0.8

    @property
    def min_request_interval(self) -> float:
        """Minimum seconds between requests: 60 / (15 * 0.8) = 5 seconds."""
        return 60.0 / (self.rpm_limit * self.rpm_buffer)


@dataclass
class LLMConfig:
    """LLM configuration for agents - model agnostic via Gemini and LiteLLM.
    
    Environment Variables (HuggingFace Space compatible):
    - LLM_PROVIDER: Provider name (gemini, openai, anthropic, groq, ollama). Default: "gemini"
    - LLM_MODEL: Model identifier. Default: "gemini-2.5-flash-lite"
    - GOOGLE_API_KEY: Gemini API key (Gemini provider)
    - GENAI_API_KEY: Alternative Gemini API key (fallback)
    - OPENAI_API_KEY: OpenAI API key (OpenAI provider)
    - ANTHROPIC_API_KEY: Anthropic API key (Anthropic provider)
    - GROQ_API_KEY: Groq API key (Groq provider)
    """
    # Provider options: gemini, openai, anthropic, groq, ollama, etc.
    # Reads from LLM_PROVIDER env var, falls back to "gemini"
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini"))
    # Model identifier - reads from LLM_MODEL env var, falls back to "gemini-2.5-flash-lite"
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gemini-2.5-flash-lite"))
    # API key - tries GOOGLE_API_KEY first (Gemini), then GENAI_API_KEY as fallback
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY"))
    temperature: float = 0.2
    max_tokens: int = 2000

    # Rate limiting
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    enable_rate_limiting: bool = True  # Set to False to disable
    
    # Multi-model fallback
    enable_fallback: bool = True  # Enable automatic model switching on rate limit
    fallback_on_error: bool = True  # Also fallback on API errors

    @property
    def model_string(self) -> str:
        """Get full model string for API calls."""
        if self.provider in ["gemini"]:
            return self.model
        else:
            # LiteLLM format: provider/model
            return f"{self.provider}/{self.model}"


@dataclass
class DatabaseConfig:
    """Database/storage configuration."""
    db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "database" / "data")
    log_path: Path = field(default_factory=lambda: PROJECT_ROOT / "database" / "logs")
    memory_path: Path = field(default_factory=lambda: PROJECT_ROOT / "database" / "memory")

    def __post_init__(self):
        # Ensure directories exist
        for path in [self.db_path, self.log_path, self.memory_path]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class CostTrackingConfig:
    """Cost and usage tracking configuration."""
    enabled: bool = True
    track_requests: bool = True
    track_tokens: bool = True
    track_time: bool = True
    
    # Usage counters (reset daily in production)
    total_requests: int = 0
    total_tokens: int = 0
    total_time_ms: float = 0.0
    
    # Per-model tracking
    model_usage: Dict[str, Dict] = field(default_factory=dict)
    
    def record_request(self, model: str, tokens: int, time_ms: float):
        """Record a request for cost tracking."""
        if not self.enabled:
            return
        
        self.total_requests += 1
        self.total_tokens += tokens
        self.total_time_ms += time_ms
        
        if model not in self.model_usage:
            self.model_usage[model] = {"requests": 0, "tokens": 0, "time_ms": 0.0}
        
        self.model_usage[model]["requests"] += 1
        self.model_usage[model]["tokens"] += tokens
        self.model_usage[model]["time_ms"] += time_ms
    
    def get_summary(self) -> Dict:
        """Get cost tracking summary."""
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_time_ms": self.total_time_ms,
            "avg_time_per_request": self.total_time_ms / max(1, self.total_requests),
            "model_breakdown": self.model_usage.copy()
        }
    
    def reset(self):
        """Reset all counters."""
        self.total_requests = 0
        self.total_tokens = 0
        self.total_time_ms = 0.0
        self.model_usage = {}


@dataclass
class EvaluationConfig:
    """Evaluation settings."""
    num_runs: int = 5  # Number of runs per problem for reliability
    timeout_seconds: float = 120.0  # Max time per problem
    save_results: bool = True
    
    # Cost tracking for evaluation
    cost_tracking: CostTrackingConfig = field(default_factory=CostTrackingConfig)


@dataclass
class SystemConfig:
    """Master configuration."""
    mcp: MCPConfig = field(default_factory=MCPConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # System mode: "blackboard", "guided", or "naked"
    active_mode: str = "guided"

    # Debug settings
    verbose: bool = True
    log_level: str = "INFO"


# Global config instance
config = SystemConfig()


def set_mode(mode: str):
    """Switch between blackboard, guided, and naked modes."""
    if mode not in ("blackboard", "guided", "naked"):
        raise ValueError(f"Invalid mode: {mode}. Use 'blackboard', 'guided', or 'naked'")
    config.active_mode = mode


def get_mode() -> str:
    """Get current system mode."""
    return config.active_mode


def set_api_key(api_key: str):
    """Set the API key for LLM calls."""
    config.llm.api_key = api_key


def get_cost_summary() -> Dict:
    """Get the current cost tracking summary."""
    return config.evaluation.cost_tracking.get_summary()


def reset_cost_tracking():
    """Reset cost tracking counters."""
    config.evaluation.cost_tracking.reset()

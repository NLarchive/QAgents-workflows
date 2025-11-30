"""
LLM Adapter: Model-agnostic LLM interface with multi-model fallback.
Supports Gemini (native), OpenAI, Anthropic, Groq, Ollama, and any LiteLLM provider.

Path: QAgents-workflos/agents/llm_adapter.py
Related: config.py (GEMINI_MODELS cascade, CostTrackingConfig)
         orchestrators/orchestrator.py (uses get_llm_adapter)
         specialized_agents.py (agents use LLM adapters)

Multi-Model Fallback System with Recovery:
==========================================
When a model hits rate limits (429) or errors, automatically falls back to next model.
RECOVERY: When preferred model cooldown expires, automatically rotates back.

Cascade order (by RPD - highest to lowest):
  1. gemma-3-27b-it (14,400 RPD) - Highest availability
  2. gemini-2.5-flash-lite (1,000 RPD) - DEFAULT PREFERRED
  3. gemini-2.5-flash (250 RPD)
  4. gemini-2.0-flash (200 RPD)
  5. gemini-2.0-flash-lite (200 RPD)
  6. gemini-2.5-pro (50 RPD) - Last resort

Model Recovery Timer:
=====================
- Tracks when each model was rate-limited
- Calculates recovery time (RPM cooldown: 60s, RPD cooldown: reset at midnight)
- Automatically returns to preferred model when recovered
- Preferred model index configurable (default: 1 = gemini-2.5-flash-lite)
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# =============================================================================
# MULTI-MODEL RATE LIMITER
# =============================================================================

class ModelRateLimiter:
    """
    Rate limiter with per-model tracking, automatic fallback, and recovery.

    Tracks:
    - RPM: Requests per minute (sliding window)
    - RPD: Requests per day (counter reset at midnight or manually)
    - Recovery: When rate-limited models become available again

    When current model exceeds limits, suggests next model in cascade.
    When preferred model recovers, automatically rotates back.
    """

    def __init__(self, models: List[Dict] = None, preferred_model_idx: int = 1):
        """
        Initialize with model cascade from config.

        Args:
            models: List of model configs with rpm, rpd limits
            preferred_model_idx: Index of preferred model (default: 1 = gemini-2.5-flash-lite)
        """
        from config import GEMINI_MODELS
        self.models = models or GEMINI_MODELS
        self.preferred_model_idx = preferred_model_idx  # Model to return to after recovery
        self.current_model_idx = preferred_model_idx  # Start with preferred model

        # Per-model tracking
        self.model_usage: Dict[str, Dict] = {}
        for model in self.models:
            self.model_usage[model["name"]] = {
                "rpm_window": deque(maxlen=model["rpm"]),  # Sliding window
                "rpd_count": 0,
                "rpd_reset_time": datetime.now().replace(hour=0, minute=0, second=0) + timedelta(days=1),
                "last_request_time": 0,
                "total_tokens": 0,
                "total_time_ms": 0.0,
                # Recovery tracking
                "rate_limited_at": None,  # Timestamp when rate limited
                "rpm_recovery_time": None,  # When RPM limit recovers
                "rpd_recovery_time": None,  # When RPD limit recovers (midnight)
            }

    @property
    def current_model(self) -> Dict:
        """Get current model config."""
        return self.models[self.current_model_idx]

    @property
    def current_model_name(self) -> str:
        """Get current model name."""
        return self.current_model["name"]

    @property
    def preferred_model_name(self) -> str:
        """Get preferred model name."""
        return self.models[self.preferred_model_idx]["name"]

    def get_min_interval(self, model_name: str = None) -> float:
        """Get minimum interval between requests for model (with 80% buffer)."""
        if model_name is None:
            model_name = self.current_model_name

        for model in self.models:
            if model["name"] == model_name:
                # 80% buffer: 60s / (rpm * 0.8)
                return 60.0 / (model["rpm"] * 0.8)
        return 5.0  # Default 5 seconds

    def check_preferred_model_recovery(self) -> bool:
        """
        Check if preferred model has recovered from rate limiting.
        If recovered, automatically switch back to it.
        
        Returns:
            True if switched back to preferred model
        """
        if self.current_model_idx == self.preferred_model_idx:
            return False  # Already on preferred model

        preferred_name = self.preferred_model_name
        usage = self.model_usage.get(preferred_name)
        if not usage:
            return False

        current_time = datetime.now()

        # Check RPD recovery (resets at midnight)
        if usage.get("rpd_recovery_time") and current_time >= usage["rpd_recovery_time"]:
            usage["rpd_count"] = 0
            usage["rpd_recovery_time"] = None
            usage["rate_limited_at"] = None
            logger.info(f"Preferred model {preferred_name} RPD limit reset - switching back")
            self.current_model_idx = self.preferred_model_idx
            return True

        # Check RPM recovery (60 seconds)
        if usage.get("rpm_recovery_time") and current_time >= usage["rpm_recovery_time"]:
            usage["rpm_recovery_time"] = None
            # Check if we can make a request now
            can_req, _ = self.can_request(preferred_name)
            if can_req:
                logger.info(f"Preferred model {preferred_name} RPM recovered - switching back")
                self.current_model_idx = self.preferred_model_idx
                return True

        return False

    def can_request(self, model_name: str = None) -> tuple[bool, str]:
        """
        Check if we can make a request with current/specified model.
        
        Returns:
            (can_request: bool, reason: str)
        """
        if model_name is None:
            model_name = self.current_model_name
        
        if model_name not in self.model_usage:
            return False, f"Unknown model: {model_name}"
        
        usage = self.model_usage[model_name]
        model_config = None
        for m in self.models:
            if m["name"] == model_name:
                model_config = m
                break
        
        if not model_config:
            return False, f"Model config not found: {model_name}"
        
        # Check RPD (reset if new day)
        if datetime.now() >= usage["rpd_reset_time"]:
            usage["rpd_count"] = 0
            usage["rpd_reset_time"] = datetime.now().replace(hour=0, minute=0, second=0) + timedelta(days=1)
        
        if usage["rpd_count"] >= model_config["rpd"]:
            return False, f"RPD limit reached ({model_config['rpd']}/day)"
        
        # Check RPM (sliding window)
        current_time = time.time()
        window = usage["rpm_window"]
        
        # Remove old entries (>60s ago)
        while window and (current_time - window[0]) > 60:
            window.popleft()
        
        if len(window) >= model_config["rpm"]:
            return False, f"RPM limit reached ({model_config['rpm']}/min)"
        
        return True, "OK"
    
    def wait_if_needed(self, model_name: str = None) -> float:
        """
        Wait if necessary to respect rate limits.
        
        Returns:
            Time waited in seconds
        """
        if model_name is None:
            model_name = self.current_model_name
        
        if model_name not in self.model_usage:
            return 0.0
        
        usage = self.model_usage[model_name]
        current_time = time.time()
        min_interval = self.get_min_interval(model_name)
        
        time_since_last = current_time - usage["last_request_time"]
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            logger.info(f"Rate limiting [{model_name}]: waiting {sleep_time:.2f}s")
            time.sleep(sleep_time)
            return sleep_time
        
        return 0.0
    
    def record_request(self, model_name: str = None, tokens: int = 0, time_ms: float = 0):
        """Record a successful request."""
        if model_name is None:
            model_name = self.current_model_name
        
        if model_name not in self.model_usage:
            return
        
        usage = self.model_usage[model_name]
        current_time = time.time()
        
        usage["rpm_window"].append(current_time)
        usage["rpd_count"] += 1
        usage["last_request_time"] = current_time
        usage["total_tokens"] += tokens
        usage["total_time_ms"] += time_ms
        
        logger.debug(f"Request recorded [{model_name}]: RPD {usage['rpd_count']}, tokens {tokens}")

    def fallback_to_next(self, reason: str = "unknown") -> Optional[str]:
        """
        Switch to next model in cascade and record recovery time.

        Args:
            reason: Why fallback is needed ("rpm", "rpd", or "error")

        Returns:
            New model name or None if no more models available
        """
        current_model_name = self.current_model_name
        usage = self.model_usage.get(current_model_name, {})
        
        # Record when this model was rate limited and set recovery time
        now = datetime.now()
        usage["rate_limited_at"] = now
        
        if "rpm" in reason.lower() or "429" in reason:
            # RPM recovery: 60 seconds from now
            usage["rpm_recovery_time"] = now + timedelta(seconds=60)
            logger.info(f"Model {current_model_name} RPM limited - recovery at {usage['rpm_recovery_time']}")
        elif "rpd" in reason.lower() or "quota" in reason.lower():
            # RPD recovery: midnight tonight
            usage["rpd_recovery_time"] = now.replace(hour=0, minute=0, second=0) + timedelta(days=1)
            logger.info(f"Model {current_model_name} RPD limited - recovery at {usage['rpd_recovery_time']}")
        
        if self.current_model_idx + 1 < len(self.models):
            self.current_model_idx += 1
            new_model = self.current_model_name
            logger.warning(f"Falling back to model: {new_model}")
            return new_model
        else:
            logger.error("No more models available in fallback cascade!")
            return None

    def reset_to_preferred(self):
        """Reset to preferred model (default: gemini-2.5-flash-lite)."""
        self.current_model_idx = self.preferred_model_idx
        logger.info(f"Reset to preferred model: {self.preferred_model_name}")

    def get_usage_summary(self) -> Dict:
        """Get usage summary for all models."""
        summary = {}
        for model in self.models:
            name = model["name"]
            usage = self.model_usage[name]
            summary[name] = {
                "rpm_used": len(usage["rpm_window"]),
                "rpm_limit": model["rpm"],
                "rpd_used": usage["rpd_count"],
                "rpd_limit": model["rpd"],
                "total_tokens": usage["total_tokens"],
                "total_time_ms": usage["total_time_ms"]
            }
        return summary


# Global rate limiter instance
_global_rate_limiter: Optional[ModelRateLimiter] = None

def get_rate_limiter() -> ModelRateLimiter:
    """Get or create global rate limiter."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = ModelRateLimiter()
    return _global_rate_limiter


# =============================================================================
# LLM RESPONSE TYPES
# =============================================================================

@dataclass
class LLMToolCall:
    """Standardized tool call across all providers."""
    tool_name: str
    arguments: Dict[str, Any]
    reasoning: str


@dataclass
class LLMResponse:
    """Standardized response across all providers."""
    text: str
    tool_calls: List[LLMToolCall]
    finish_reason: str
    model_used: str = ""  # Track which model was actually used
    tokens_used: int = 0  # Track token usage if available
    time_ms: float = 0.0  # Track response time


# =============================================================================
# BASE ADAPTER
# =============================================================================

class BaseLLMAdapter(ABC):
    """Abstract base for LLM adapters."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    @abstractmethod
    def generate(self,
                 messages: List[Dict[str, str]],
                 tools: Optional[List[Dict[str, Any]]] = None,
                 temperature: float = 0.2,
                 max_tokens: int = 2000) -> LLMResponse:
        """Generate a response from the LLM."""
        pass


# =============================================================================
# GEMINI ADAPTER WITH FALLBACK
# =============================================================================

class GeminiAdapter(BaseLLMAdapter):
    """
    Google Gemini API adapter with multi-model fallback.
    
    Automatically falls back to next model when:
    - Rate limit exceeded (429)
    - API error occurs (if fallback_on_error=True)
    - Model unavailable
    """

    def __init__(self, 
                 model: str = "gemini-2.5-flash-lite", 
                 api_key: Optional[str] = None,
                 enable_fallback: bool = True):
        super().__init__(api_key)
        self.model = model
        self.enable_fallback = enable_fallback
        self._client = None
        self.rate_limiter = get_rate_limiter()

    def _get_client(self):
        """Lazy load Gemini client."""
        if self._client is None:
            try:
                import google.genai
                self._client = google.genai.Client(api_key=self.api_key)
            except ImportError:
                raise ImportError("google-genai not installed. Install with: pip install google-genai")
        return self._client

    def generate(self,
                 messages: List[Dict[str, str]],
                 tools: Optional[List[Dict[str, Any]]] = None,
                 temperature: float = 0.2,
                 max_tokens: int = 2000) -> LLMResponse:
        """
        Generate content using Gemini with automatic fallback.
        
        Will try current model first, then fall back through cascade on errors.
        """
        start_time = time.time()
        last_error = None
        attempts = 0
        max_attempts = len(self.rate_limiter.models)
        
        while attempts < max_attempts:
            current_model = self.rate_limiter.current_model_name
            attempts += 1
            
            try:
                # Check if preferred model has recovered
                self.rate_limiter.check_preferred_model_recovery()

                # Check if we can make a request
                can_request, reason = self.rate_limiter.can_request(current_model)

                if not can_request:
                    logger.warning(f"Cannot request from {current_model}: {reason}")
                    if self.enable_fallback:
                        next_model = self.rate_limiter.fallback_to_next(reason)
                        if next_model:
                            continue
                    raise Exception(f"Rate limit exceeded: {reason}")                # Wait if needed for RPM
                self.rate_limiter.wait_if_needed(current_model)
                
                # Make the actual API call
                response = self._call_gemini(current_model, messages, tools, temperature, max_tokens)
                
                # Record successful request
                elapsed_ms = (time.time() - start_time) * 1000
                tokens = self._estimate_tokens(messages, response.text)
                self.rate_limiter.record_request(current_model, tokens, elapsed_ms)
                
                # Update response metadata
                response.model_used = current_model
                response.tokens_used = tokens
                response.time_ms = elapsed_ms
                
                # Record in global cost tracking
                try:
                    from config import config
                    config.evaluation.cost_tracking.record_request(current_model, tokens, elapsed_ms)
                except Exception:
                    pass  # Config might not be available
                
                return response
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Check if it's a rate limit error
                is_rate_limit = "429" in str(e) or "rate" in error_str or "quota" in error_str
                
                if is_rate_limit or (self.enable_fallback and "error" in error_str):
                    logger.warning(f"Error with {current_model}: {e}")
                    next_model = self.rate_limiter.fallback_to_next(error_str)
                    if next_model:
                        logger.info(f"Retrying with fallback model: {next_model}")
                        continue
                
                # Non-recoverable error or no fallback
                raise
        
        # Exhausted all models
        raise Exception(f"All models exhausted. Last error: {last_error}")
    
    def _call_gemini(self,
                     model: str,
                     messages: List[Dict[str, str]],
                     tools: Optional[List[Dict[str, Any]]],
                     temperature: float,
                     max_tokens: int) -> LLMResponse:
        """Make actual Gemini API call."""
        client = self._get_client()
        
        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            role = "user" if msg["role"] in ["user", "system"] else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
        
        # Build tools for Gemini
        gemini_tools = None
        if tools:
            gemini_tools = [{
                "function_declarations": [t["function"] for t in tools]
            }]
        
        # Call Gemini - tools go in config
        config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens
        }
        if gemini_tools:
            config["tools"] = gemini_tools
        
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )
        
        # Extract response
        text = response.text if hasattr(response, 'text') and response.text else ""
        tool_calls = []

        if hasattr(response, 'function_calls') and response.function_calls:
            for func_call in response.function_calls:
                args = func_call.args if isinstance(func_call.args, dict) else json.loads(str(func_call.args))
                tool_calls.append(LLMToolCall(
                    tool_name=func_call.name,
                    arguments=args,
                    reasoning=text or "Tool selected by Gemini"
                ))

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            finish_reason=str(response.finish_reason) if hasattr(response, 'finish_reason') else "STOP"
        )
    
    def _estimate_tokens(self, messages: List[Dict], response_text: str) -> int:
        """Estimate token count (rough: 4 chars = 1 token)."""
        input_chars = sum(len(m.get("content", "") or "") for m in messages)
        output_chars = len(response_text or "")
        return (input_chars + output_chars) // 4


# =============================================================================
# LITELLM ADAPTER
# =============================================================================

class LiteLLMAdapter(BaseLLMAdapter):
    """LiteLLM adapter for OpenAI, Anthropic, Groq, Ollama, and others."""

    def __init__(self, model: str = "gpt-4o-mini", provider: str = "openai", api_key: Optional[str] = None):
        super().__init__(api_key)
        self.provider = provider
        self.model_string = f"{provider}/{model}" if provider else model
        self._client = None

    def _get_client(self):
        """Lazy load LiteLLM client."""
        if self._client is None:
            try:
                import litellm
                if self.api_key:
                    litellm.api_key = self.api_key
                self._client = litellm
            except ImportError:
                raise ImportError("litellm not installed. Install with: pip install litellm")
        return self._client

    def generate(self,
                 messages: List[Dict[str, str]],
                 tools: Optional[List[Dict[str, Any]]] = None,
                 temperature: float = 0.2,
                 max_tokens: int = 2000) -> LLMResponse:
        """Generate content using LiteLLM."""
        try:
            start_time = time.time()
            client = self._get_client()

            # Call LiteLLM
            response = client.completion(
                model=self.model_string,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Extract response
            choice = response.choices[0]
            text = choice.message.content or ""
            tool_calls = []

            if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    args = json.loads(tool_call.function.arguments)
                    tool_calls.append(LLMToolCall(
                        tool_name=tool_call.function.name,
                        arguments=args,
                        reasoning=text or "Tool selected by LLM"
                    ))

            elapsed_ms = (time.time() - start_time) * 1000
            tokens = response.usage.total_tokens if hasattr(response, 'usage') else 0
            
            return LLMResponse(
                text=text,
                tool_calls=tool_calls,
                finish_reason=choice.finish_reason,
                model_used=self.model_string,
                tokens_used=tokens,
                time_ms=elapsed_ms
            )

        except Exception as e:
            logger.error(f"LiteLLM generation failed: {e}")
            raise


# =============================================================================
# MOCK ADAPTER FOR TESTING
# =============================================================================

class MockLLMAdapter(BaseLLMAdapter):
    """Mock LLM for testing without API keys."""

    def generate(self,
                 messages: List[Dict[str, str]],
                 tools: Optional[List[Dict[str, Any]]] = None,
                 temperature: float = 0.2,
                 max_tokens: int = 2000) -> LLMResponse:
        """Return a mock response."""
        return LLMResponse(
            text="Mock LLM response",
            tool_calls=[],
            finish_reason="stop",
            model_used="mock",
            tokens_used=10,
            time_ms=1.0
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_llm_adapter(provider: str = "gemini",
                    model: str = "gemini-2.5-flash-lite",
                    api_key: Optional[str] = None,
                    enable_fallback: bool = True) -> BaseLLMAdapter:
    """
    Factory function to get the appropriate LLM adapter.
    
    Args:
        provider: LLM provider (gemini, openai, anthropic, etc.)
        model: Model name
        api_key: API key for authentication
        enable_fallback: Enable automatic model fallback on rate limits
    
    Returns:
        Configured LLM adapter
    """
    if provider == "gemini":
        try:
            return GeminiAdapter(model=model, api_key=api_key, enable_fallback=enable_fallback)
        except ImportError:
            logger.warning("Gemini not available, trying LiteLLM")
            return LiteLLMAdapter(model=model, provider="gemini", api_key=api_key)

    elif provider in ["openai", "anthropic", "groq", "ollama", "cohere", "mistral"]:
        return LiteLLMAdapter(model=model, provider=provider, api_key=api_key)

    elif provider == "mock":
        return MockLLMAdapter(api_key=api_key)

    else:
        # Try LiteLLM for unknown providers
        logger.warning(f"Unknown provider {provider}, attempting LiteLLM")
        return LiteLLMAdapter(model=model, provider=provider, api_key=api_key)


def get_usage_summary() -> Dict:
    """Get usage summary from global rate limiter."""
    return get_rate_limiter().get_usage_summary()


def reset_rate_limiter():
    """Reset rate limiter to default state."""
    global _global_rate_limiter
    _global_rate_limiter = None

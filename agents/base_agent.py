"""
Agents Module: Base agent classes and specialized agents.
Supports both Blackboard (free) and Guided (strict) architectures.
Model-agnostic: Works with Gemini, OpenAI, Anthropic, Groq, Ollama, etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Roles agents can take in the system."""
    ARCHITECT = "architect"
    BUILDER = "builder"
    VALIDATOR = "validator"
    OPTIMIZER = "optimizer"
    ANALYZER = "analyzer"
    SCORER = "scorer"
    COORDINATOR = "coordinator"


class AgentState(Enum):
    """Agent execution states."""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentContext:
    """Context passed to agents for decision making."""
    goal: str
    current_circuit: Optional[str] = None
    history: List[Dict] = field(default_factory=list)
    constraints: Dict = field(default_factory=dict)
    shared_data: Dict = field(default_factory=dict)
    
    def add_to_history(self, action: str, result: Any):
        self.history.append({
            "action": action,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })


@dataclass
class AgentAction:
    """An action an agent wants to take."""
    tool_name: str
    arguments: Dict
    reasoning: str
    priority: float = 1.0


@dataclass
class AgentResult:
    """Result of an agent's execution."""
    success: bool
    data: Any
    message: str
    actions_taken: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    Provides common interface for both Blackboard and Guided architectures.
    """
    
    def __init__(self,
                 agent_id: str,
                 role: AgentRole,
                 tools: List[str] = None,
                 llm_config: Dict = None):
        self.agent_id = agent_id
        self.role = role
        self.tools = tools or []
        self.llm_config = llm_config or {}
        self.state = AgentState.IDLE
        self.memory: Dict = {}
        self._callbacks: List[Callable] = []
        
    @abstractmethod
    def decide(self, context: AgentContext) -> Optional[AgentAction]:
        """Decide what action to take given the context."""
        pass
        
    @abstractmethod
    def execute(self, action: AgentAction, context: AgentContext) -> AgentResult:
        """Execute the decided action."""
        pass
        
    def can_handle(self, context: AgentContext) -> bool:
        """Check if this agent can handle the current context."""
        return True
        
    def on_state_change(self, callback: Callable):
        """Register callback for state changes."""
        self._callbacks.append(callback)
        
    def _set_state(self, new_state: AgentState):
        """Update state and notify callbacks."""
        old_state = self.state
        self.state = new_state
        for cb in self._callbacks:
            cb(self.agent_id, old_state, new_state)
            
    def reset(self):
        """Reset agent to initial state."""
        self.state = AgentState.IDLE
        self.memory.clear()


class LLMAgent(BaseAgent):
    """
    Agent that uses an LLM for decision making.
    Model-agnostic: Supports Gemini, OpenAI, Anthropic, Groq, Ollama, etc.
    Can be used in both Blackboard and Guided modes.
    """
    
    def __init__(self,
                 agent_id: str,
                 role: AgentRole,
                 system_prompt: str,
                 tools: List[str] = None,
                 llm_config: Dict = None):
        super().__init__(agent_id, role, tools, llm_config)
        self.system_prompt = system_prompt
        self._adapter = None
        
    def _get_adapter(self):
        """Get the LLM adapter (lazy init)."""
        if self._adapter is None:
            from config import config
            from agents.llm_adapter import get_llm_adapter
            
            self._adapter = get_llm_adapter(
                provider=config.llm.provider,
                model=config.llm.model,
                api_key=config.llm.api_key
            )
        return self._adapter
        
    def _build_messages(self, context: AgentContext) -> List[Dict]:
        """Build message list for LLM."""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        context_msg = f"""
Goal: {context.goal}

Current Circuit:
{context.current_circuit or 'None yet'}

Constraints:
{json.dumps(context.constraints, indent=2)}

History (last 5 actions):
{json.dumps(context.history[-5:], indent=2)}
"""
        messages.append({"role": "user", "content": context_msg})
        return messages
        
    def decide(self, context: AgentContext) -> Optional[AgentAction]:
        """Use LLM to decide on action."""
        self._set_state(AgentState.THINKING)
        
        try:
            from config import config
            from tools import registry
            
            tool_schemas = [
                registry.get(name).to_llm_schema()
                for name in self.tools
                if registry.get(name)
            ]
            
            messages = self._build_messages(context)
            adapter = self._get_adapter()
            
            llm_response = adapter.generate(
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
                temperature=self.llm_config.get("temperature", config.llm.temperature),
                max_tokens=self.llm_config.get("max_tokens", config.llm.max_tokens)
            )
            
            if llm_response.tool_calls:
                tool_call = llm_response.tool_calls[0]
                return AgentAction(
                    tool_name=tool_call.tool_name,
                    arguments=tool_call.arguments,
                    reasoning=tool_call.reasoning
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} decision failed: {e}")
            self._set_state(AgentState.ERROR)
            return None
            
    def execute(self, action: AgentAction, context: AgentContext) -> AgentResult:
        """Execute tool action."""
        self._set_state(AgentState.EXECUTING)
        
        import time
        start = time.perf_counter()
        
        try:
            from tools import invoke_tool
            
            result = invoke_tool(action.tool_name, **action.arguments)
            elapsed = (time.perf_counter() - start) * 1000
            
            context.add_to_history(action.tool_name, result)
            
            self._set_state(AgentState.COMPLETED)
            return AgentResult(
                success=result.get("success", False),
                data=result,
                message=f"Executed {action.tool_name}",
                actions_taken=[action.tool_name],
                execution_time_ms=elapsed
            )
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} execution failed: {e}")
            self._set_state(AgentState.ERROR)
            return AgentResult(
                success=False,
                data=None,
                message=str(e)
            )


class RuleBasedAgent(BaseAgent):
    """
    Agent that uses predefined rules for decision making.
    Useful for deterministic behavior in Guided mode.
    """
    
    def __init__(self,
                 agent_id: str,
                 role: AgentRole,
                 rules: List[Callable[[AgentContext], Optional[AgentAction]]],
                 tools: List[str] = None):
        super().__init__(agent_id, role, tools)
        self.rules = rules
        
    def decide(self, context: AgentContext) -> Optional[AgentAction]:
        """Apply rules to decide action."""
        self._set_state(AgentState.THINKING)
        
        for rule in self.rules:
            action = rule(context)
            if action is not None:
                return action
                
        return None
        
    def execute(self, action: AgentAction, context: AgentContext) -> AgentResult:
        """Execute action using tools."""
        self._set_state(AgentState.EXECUTING)
        
        import time
        start = time.perf_counter()
        
        try:
            from tools import invoke_tool
            
            result = invoke_tool(action.tool_name, **action.arguments)
            elapsed = (time.perf_counter() - start) * 1000
            
            context.add_to_history(action.tool_name, result)
            
            self._set_state(AgentState.COMPLETED)
            return AgentResult(
                success=result.get("success", False),
                data=result,
                message=f"Executed {action.tool_name}",
                actions_taken=[action.tool_name],
                execution_time_ms=elapsed
            )
            
        except Exception as e:
            self._set_state(AgentState.ERROR)
            return AgentResult(
                success=False,
                data=None,
                message=str(e)
            )

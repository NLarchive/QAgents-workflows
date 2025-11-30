"""
Tools Module: Wrapped MCP endpoints as callable tools for agents.
Each tool is a self-contained function that can be invoked by agents.
"""

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import json

class ToolCategory(Enum):
    """Categories of tools for agent specialization."""
    CREATION = "creation"
    ANALYSIS = "analysis" 
    VALIDATION = "validation"
    SIMULATION = "simulation"
    SCORING = "scoring"
    COMPOSITION = "composition"
    RESOURCE = "resource"

@dataclass
class ToolDefinition:
    """Definition of a tool that agents can use."""
    name: str
    description: str
    category: ToolCategory
    parameters: Dict[str, Dict]  # name -> {type, description, required}
    function: Callable
    returns: str
    
    def to_llm_schema(self) -> Dict:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []
        
        for name, info in self.parameters.items():
            properties[name] = {
                "type": info.get("type", "string"),
                "description": info.get("description", "")
            }
            if info.get("required", False):
                required.append(name)
                
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }


class ToolRegistry:
    """Registry of all available tools."""
    
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._by_category: Dict[ToolCategory, List[str]] = {cat: [] for cat in ToolCategory}
        
    def register(self, tool: ToolDefinition):
        """Register a tool."""
        self._tools[tool.name] = tool
        self._by_category[tool.category].append(tool.name)
        
    def get(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self._tools.get(name)
        
    def get_by_category(self, category: ToolCategory) -> List[ToolDefinition]:
        """Get all tools in a category."""
        return [self._tools[name] for name in self._by_category[category]]
        
    def get_all(self) -> List[ToolDefinition]:
        """Get all registered tools."""
        return list(self._tools.values())
        
    def get_llm_schemas(self, categories: Optional[List[ToolCategory]] = None) -> List[Dict]:
        """Get OpenAI function schemas for specified categories."""
        if categories is None:
            tools = self.get_all()
        else:
            tools = []
            for cat in categories:
                tools.extend(self.get_by_category(cat))
        return [t.to_llm_schema() for t in tools]
        
    def invoke(self, name: str, **kwargs) -> Any:
        """Invoke a tool by name with arguments."""
        tool = self.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")
        return tool.function(**kwargs)


# Global registry
registry = ToolRegistry()


def register_tool(name: str, description: str, category: ToolCategory,
                  parameters: Dict, returns: str):
    """Decorator to register a function as a tool."""
    def decorator(func: Callable):
        tool = ToolDefinition(
            name=name,
            description=description,
            category=category,
            parameters=parameters,
            function=func,
            returns=returns
        )
        registry.register(tool)
        return func
    return decorator

"""
QAgents-Workflows: Hugging Face Space Entry Point
Provides a Gradio interface for the Quantum Circuit Orchestrator.
Reads all configuration from environment variables for HF Space deployment.
"""

import os
import gradio as gr
import logging
from config import LLMConfig
from orchestrators import create_orchestrator
from client.mcp_client import get_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log environment configuration at startup
logger.info("=" * 70)
logger.info("QAgents Quantum Circuit Orchestrator - Initialization")
logger.info("=" * 70)
logger.info(f"LLM Provider: {os.getenv('LLM_PROVIDER', 'gemini (default)')}")
logger.info(f"LLM Model: {os.getenv('LLM_MODEL', 'gemini-2.5-flash-lite (default)')}")
logger.info(f"MCP Server URL: {os.getenv('MCP_SERVER_URL', 'http://127.0.0.1:7861 (default)')}")
logger.info(f"Google API Key configured: {bool(os.getenv('GOOGLE_API_KEY') or os.getenv('GENAI_API_KEY'))}")
logger.info("=" * 70)

# Initialize MCP client (will use MCP_SERVER_URL env var if set)
mcp_client = get_client()

def generate_circuit(prompt, mode, difficulty):
    """Generate a quantum circuit based on the prompt and mode."""
    try:
        logger.info(f"Generating circuit: mode={mode}, difficulty={difficulty}")
        logger.info(f"Prompt: {prompt}")
        
        # Create orchestrator
        orch = create_orchestrator(mode.lower())
        
        # Run generation
        # Note: In a real deployment, we might want to map difficulty to specific constraints
        # For now, we pass the prompt directly
        result = orch.run(prompt)
        
        if result.success:
            output = f"✅ Success ({result.execution_time_ms:.0f}ms)\n\n"
            if result.final_output:
                output += result.final_output
            else:
                output += "No QASM generated."
                
            # Add metrics if available
            metrics = f"LLM Calls: {result.steps_completed}\n"
            if hasattr(result, 'tokens_used'):
                metrics += f"Tokens: {result.tokens_used}\n"
                
            return output, metrics
        else:
            error_msg = "\n".join(result.errors)
            return f"❌ Failed ({result.execution_time_ms:.0f}ms)\n\nErrors:\n{error_msg}", "N/A"
            
    except Exception as e:
        logger.error(f"Error generating circuit: {e}")
        return f"❌ System Error: {str(e)}", "Error"

def check_mcp_status():
    """Check connection to MCP server."""
    try:
        is_healthy = mcp_client.health_check()
        status = "🟢 Connected" if is_healthy else "🔴 Disconnected"
        url = os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:7861")
        return f"{status} ({url})"
    except Exception as e:
        return f"🔴 Error: {str(e)}"

# Create Gradio Interface
with gr.Blocks(title="Quantum Circuit Orchestrator") as demo:
    gr.Markdown("# ⚛️ QAgents: Quantum Circuit Orchestrator")
    gr.Markdown("Multi-agent system for generating optimized quantum circuits.")
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="Circuit Description",
                placeholder="e.g., Create a 3-qubit GHZ state",
                lines=3
            )
            with gr.Row():
                mode_select = gr.Dropdown(
                    choices=["naked", "quasar", "hybrid", "blackboard"],
                    value="naked",
                    label="Orchestration Mode"
                )
                difficulty_select = gr.Dropdown(
                    choices=["EASY", "MEDIUM", "HARD", "VERY_HARD"],
                    value="EASY",
                    label="Estimated Difficulty"
                )
            
            generate_btn = gr.Button("Generate Circuit", variant="primary")
            
        with gr.Column(scale=1):
            mcp_status = gr.Textbox(label="MCP Server Status", value=check_mcp_status, interactive=False)
            metrics_output = gr.Textbox(label="Execution Metrics", lines=4)
    
    with gr.Row():
        qasm_output = gr.Code(label="Generated QASM", language="qasm", lines=15)

    # Event handlers
    generate_btn.click(
        fn=generate_circuit,
        inputs=[prompt_input, mode_select, difficulty_select],
        outputs=[qasm_output, metrics_output]
    )
    
    # Refresh status on load
    demo.load(fn=check_mcp_status, outputs=[mcp_status])

if __name__ == "__main__":
    demo.launch()

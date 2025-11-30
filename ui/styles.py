# Path: QAgents-workflows/ui/styles.py
# Relations: Used by __init__.py, app.py
# Description: Custom CSS styles for the Gradio app interface
"""
Styles Module: Custom CSS for QAgents Gradio interface.
Provides consistent styling across all UI components.
"""

CUSTOM_CSS = """
/* Main container styling */
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}

/* Header styling */
.app-header {
    text-align: center;
    padding: 1rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
    margin-bottom: 1rem;
    color: white;
}

/* Chat container */
.chat-container {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 1rem;
}

/* Message styling */
.user-message {
    background-color: #e3f2fd;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    margin: 0.25rem 0;
}

.bot-message {
    background-color: #f5f5f5;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    margin: 0.25rem 0;
}

/* Code blocks in chat */
.code-block {
    background-color: #1e1e1e;
    color: #d4d4d4;
    padding: 1rem;
    border-radius: 5px;
    font-family: 'Fira Code', 'Consolas', monospace;
    overflow-x: auto;
}

/* Status indicators */
.status-connected {
    color: #4caf50;
    font-weight: bold;
}

.status-disconnected {
    color: #f44336;
    font-weight: bold;
}

.status-partial {
    color: #ff9800;
    font-weight: bold;
}

/* Health check table */
.health-table {
    width: 100%;
    border-collapse: collapse;
}

.health-table th, .health-table td {
    padding: 0.5rem;
    text-align: left;
    border-bottom: 1px solid #e0e0e0;
}

/* Button styling */
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
}

.primary-btn:hover {
    opacity: 0.9;
}

/* Tab styling */
.tab-nav {
    border-bottom: 2px solid #667eea;
}

/* Circuit output box */
.circuit-output {
    font-family: 'Fira Code', 'Consolas', monospace;
    background-color: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 5px;
    padding: 1rem;
}

/* Loading indicator */
.loading {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(102, 126, 234, 0.3);
    border-radius: 50%;
    border-top-color: #667eea;
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .gradio-container {
        padding: 0.5rem !important;
    }
    
    .app-header h1 {
        font-size: 1.5rem;
    }
}
"""

# Additional component-specific styles
CHAT_STYLES = """
.chatbot-container {
    min-height: 400px;
    max-height: 600px;
}
"""

MCP_HEALTH_STYLES = """
.endpoint-card {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}
"""

QUICK_BUILD_STYLES = """
.template-selector {
    margin-bottom: 1rem;
}

.qubit-slider {
    margin: 1rem 0;
}
"""

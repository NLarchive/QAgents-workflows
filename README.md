---
title: QAgents Quantum Circuit Orchestrator
emoji: ⚛️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
short_description: Multi-agent quantum circuit generation with Gemini/LLMs
---

# QAgents-Workflows: Quantum Circuit Optimization Agent System

A professional multi-agent system for autonomous quantum circuit optimization, featuring multiple architectural approaches and **model-agnostic LLM support** (Gemini, OpenAI, Anthropic, Groq, Ollama, and any LiteLLM provider).

## 🏗️ Architectures

### 1. Blackboard System (Free/Emergent)
- Agents communicate through a shared blackboard
- Decoupled, event-driven activation
- Emergent workflow based on data availability
- Maximum flexibility and adaptability

### 2. Guided System (Strict Orchestration)
- Explicit state machine with defined transitions
- Central orchestrator controls workflow
- Predictable, auditable execution path
- Maximum reliability and control

### 3. Naked System (Baseline)
- Single agent with direct MCP access
- No framework overhead
- Baseline for comparison

## 🤖 Model-Agnostic LLM Support

The system works with **any LLM provider**:

| Provider | Setup | Models |
|----------|-------|--------|
| **Gemini** (Default) | `GOOGLE_API_KEY` | `gemini-2.5-flash-lite` |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o`, `gpt-4o-mini` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-3-opus`, `claude-3-sonnet` |
| Groq | `GROQ_API_KEY` | `llama-3-70b`, `mixtral-8x7b` |
| Ollama (Local) | No key needed | Any local model |

**See [SETUP.md](SETUP.md) for detailed configuration.**

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Time** | Total execution time in seconds |
| **Quality** | Circuit depth, gate count, hardware fitness score |
| **Effectiveness** | Did the circuit achieve the goal? |
| **Reliability** | Success rate across multiple runs |

## 🚀 Quick Start

```bash
# 1. Ensure QuantumArchitect-MCP is running
python QuantumArchitect-MCP/app.py

# 2. Set your API key (for Gemini by default)
set GOOGLE_API_KEY=your-key-here
# OR for OpenAI:
set OPENAI_API_KEY=your-key-here

# 3. Run the evaluation
python QAgents-workflos/run_evaluation.py

# For quick test (no LLM needed):
python QAgents-workflos/run_evaluation.py --quick

# Test specific mode:
python QAgents-workflos/run_evaluation.py --mode guided
python QAgents-workflos/run_evaluation.py --mode blackboard
python QAgents-workflos/run_evaluation.py --mode naked
```

## 🔧 Switching LLM Providers

### Using Gemini (Default)
```bash
set GOOGLE_API_KEY=your-gemini-key
# Models: gemini-2.5-flash-lite, gemini-2.5-pro
```

### Using OpenAI
Edit `config.py`:
```python
provider: str = "openai"
model: str = "gpt-4o-mini"
```
```bash
set OPENAI_API_KEY=sk-...
```

### Using Anthropic
```python
provider: str = "anthropic"
model: str = "claude-3-sonnet-20240229"
```
```bash
set ANTHROPIC_API_KEY=your-key
```

### Using Groq
```python
provider: str = "groq"
model: str = "llama-3-70b-versatile"
```
```bash
set GROQ_API_KEY=your-key
```

### Using Local Ollama
```python
provider: str = "ollama"
model: str = "mistral"
```
No API key needed - runs locally on `http://localhost:11434`

## 📁 Project Structure

```
QAgents-workflos/
├── agents/                    # Agent implementations (Architect, Builder, etc.)
├── client/                    # MCP client for QuantumArchitect-MCP
├── database/                  # Storage layer (logs, memory, circuits)
├── orchestrators/             # Orchestration modes (Naked, Guided, Blackboard, QUASAR, Hybrid)
├── prompts/                   # System prompts for agents and optimized LLM prompts
├── tools/                     # Tool registry and MCP endpoint wrappers
├── workflows/                 # Workflow definitions
├── tests/                     # Evaluation harnesses and test problems
├── app.py                     # Gradio UI entry point (Hugging Face Space)
├── config.py                  # Configuration with env var support
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
└── README.md                  # This file
```

## 🚀 Deployment to Hugging Face Spaces

### Prerequisites
1. Create a Hugging Face Space: https://huggingface.co/new-space
2. Select **Gradio** as the SDK
3. Push this repository to your Space

### Environment Variables Configuration

The system reads configuration from **environment variables**, making it compatible with Hugging Face Spaces.

#### Critical Variables

| Variable | Purpose | Default | Example |
|----------|---------|---------|---------|
| `LLM_PROVIDER` | LLM provider to use | `gemini` | `gemini`, `openai`, `anthropic` |
| `LLM_MODEL` | Model identifier | `gemini-2.5-flash-lite` | `gpt-4o-mini`, `claude-3-sonnet` |
| `GOOGLE_API_KEY` | Gemini API key | (none) | Your API key from aistudio.google.com |
| `MCP_SERVER_URL` | Backend URL | `http://127.0.0.1:7861` | `https://your-backend.ngrok.io` |

#### Setting Variables in Hugging Face Space

**Option 1: Via Space Settings (Recommended)**
1. Go to your Space settings
2. Click **"Secrets and variables"** > **"New secret"**
3. Add each variable:
   - **Secret Name**: `GOOGLE_API_KEY` | **Value**: Your API key
   - **Secret Name**: `MCP_SERVER_URL` | **Value**: Backend URL
4. Add variables (non-sensitive):
   - **Variable Name**: `LLM_PROVIDER` | **Value**: `gemini`
   - **Variable Name**: `LLM_MODEL` | **Value**: `gemini-2.5-flash-lite`

**Option 2: Via .env File**
```bash
# Copy .env.example to .env and fill in values
cp .env.example .env

# Commit and push to your Space
git add .env
git commit -m "Add environment configuration"
git push
```

**⚠️ Important**: Never commit sensitive API keys directly. Use Space Secrets instead.

### LLM Provider Configuration

#### Using Gemini (Default)
```
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash-lite
GOOGLE_API_KEY=your-key-from-https://aistudio.google.com/app/apikey
```

#### Using OpenAI
```
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

#### Using Anthropic
```
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-sonnet-20240229
ANTHROPIC_API_KEY=sk-ant-...
```

#### Using Groq
```
LLM_PROVIDER=groq
LLM_MODEL=llama-3-70b-versatile
GROQ_API_KEY=gsk_...
```

#### Using Local Ollama
```
LLM_PROVIDER=ollama
LLM_MODEL=mistral
# No API key needed - runs locally on http://localhost:11434
```

### Backend Connection (MCP Server)

The Space communicates with the QuantumArchitect-MCP backend via `MCP_SERVER_URL`.

**Options:**

1. **Local Development** (both running on your machine):
   ```
   MCP_SERVER_URL=http://127.0.0.1:7861
   ```

2. **Public Backend with ngrok** (tunnel remote server):
   ```bash
   # On your backend server:
   ngrok http 7861
   ```
   Then set:
   ```
   MCP_SERVER_URL=https://your-ngrok-url.ngrok.io
   ```

3. **Deployed Backend** (your own server):
   ```
   MCP_SERVER_URL=https://your-quantum-api.example.com
   ```

If `MCP_SERVER_URL` is not set or unreachable, the Space will still work but with local-only features.

## 📁 Project Structure (Previous)
├── agents/           # Agent definitions (types, behaviors)
├── prompts/          # System prompts for each agent
├── tools/            # MCP tool wrappers
├── workflows/        # Workflow definitions
├── orchestrators/    # Workflow orchestration logic
├── client/           # MCP client connection
├── database/         # Memory, logs, results storage
├── tests/            # Evaluation framework
└── config.py         # Global configuration
```

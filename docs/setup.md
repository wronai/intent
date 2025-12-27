# Setup Guide

## Requirements

- **Python**: 3.10+
- **Docker**: For running the vector database (optional, depending on config) and secure sandbox.
- **LLM Provider**:
  - **Ollama** (Recommended for local dev): [Download Ollama](https://ollama.com) and pull models like `llama3.1:8b`.
  - **Anthropic/OpenAI**: API key required.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/wronai/intent.git
   cd intent
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install fastapi uvicorn python-multipart  # For server examples
   ```

## Configuration

IntentForge is configured via environment variables or a `.env` file.

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM Backend (`ollama`, `anthropic`, `openai`) | `ollama` |
| `LLM_MODEL` | Model to use (e.g., `llama3.1:8b`) | `llama3.1:8b` |
| `OLLAMA_HOST` | URL of Ollama server | `http://localhost:11434` |
| `INTENT_SANDBOX` | Enable secure code execution sandbox | `True` |

### Example `.env` for Local Development

```bash
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
OLLAMA_HOST=http://localhost:11434
INTENT_SANDBOX=True
```

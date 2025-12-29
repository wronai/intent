# Setup Guide

**[🏠 Home](INDEX.md) | [⚙️ Setup](setup.md) | [🛠️ Usage](usage.md) | [🏗️ Architecture](architecture.md) | [📚 API](api.md) | [🔧 Services](services.md) | [📝 DSL](dsl.md)**

## Requirements

- **Python**: 3.10+
- **Docker**: Recommended for full deployment
- **LLM Provider**:
  - **Ollama** (Recommended): [Download Ollama](https://ollama.com) and pull models like `llama3.1:8b`
  - **Anthropic/OpenAI**: API key required

## Quick Start (Docker)

```bash
# Clone repository
git clone https://github.com/wronai/intent.git
cd intent

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start all services
docker-compose up -d

# Open browser
open http://localhost/examples/usecases/
```

## Manual Installation

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
   pip install -e ".[server]"
   ```

4. **Run server**:
   ```bash
   python examples/server.py
   ```

## Configuration

IntentForge is configured via environment variables or a `.env` file.

### Core Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM Backend (`ollama`, `anthropic`, `openai`) | `ollama` |
| `LLM_MODEL` | Model to use (e.g., `llama3.1:8b`) | `llama3.1:8b` |
| `OLLAMA_HOST` | URL of Ollama server | `http://localhost:11434` |

### Vision Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `VISION_PROVIDER` | Vision provider | `ollama` |
| `VISION_MODEL` | Vision model (e.g., `llava:13b`) | `llava:13b` |
| `VISION_MAX_TOKENS` | Max tokens for vision | `2048` |

### Example `.env`

```bash
# LLM Configuration
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
OLLAMA_HOST=http://localhost:11434

# Vision Configuration
VISION_PROVIDER=ollama
VISION_MODEL=llava:13b
VISION_MAX_TOKENS=2048
```

## Verify Installation

```bash
# Check services
intentforge services

# Test chat
intentforge dsl-call chat send '{"message": "Hello!"}'

# Interactive REPL
intentforge repl
```

# IntentForge Documentation

**[🏠 Home](INDEX.md) | [⚙️ Setup](setup.md) | [🛠️ Usage](usage.md) | [🏗️ Architecture](ARCHITECTURE.md) | [📚 API](api.md) | [🔧 Services](services.md) | [📝 DSL](dsl.md)**

> **Version**: 0.3.0-dev | **Tests**: 20 passing | **Coverage**: 12%

Welcome to the **IntentForge** documentation. IntentForge is a framework for **Intent-Driven Development**, allowing you to generate and execute code dynamically from natural language descriptions.

## Key Features

### Core
- **🤖 LLM Integration** - Ollama, Anthropic, OpenAI, LiteLLM support
- **👁️ Vision AI** - Image analysis with LLaVA for OCR and object detection
- **📄 Document Processing** - Two-phase OCR pipeline (Tesseract + Vision + LLM)
- **📝 DSL** - Domain Specific Language for scripting and automation

### Auto-Fix & Self-Healing
- **🔧 Code Runner** - Auto-fix execution with dependency installation
- **🧪 Code Tester** - TDD with auto-generated tests and fix loop
- **🌳 Conversation Engine** - Branching conversations for parallel error resolution
- **📊 Git Tracking** - Track iterations as commits for LLM context

### Services
- **💬 Chat** - Full chat interface with conversation history
- **🎤 Voice** - NLP-powered voice command parsing
- **📈 Analytics** - LLM-generated analytics and natural language queries
- **🔌 REST API** - Generic service endpoints for all features

## Navigation

- **[Setup Guide](setup.md)**: Installation, requirements, and configuration
- **[Usage Guide](usage.md)**: How to use the Frontend and Backend API
- **[Architecture](architecture.md)**: System overview, components, and diagrams
- **[API Reference](api.md)**: REST API documentation
- **[Services](services.md)**: Available services (chat, analytics, voice, file, etc.)
- **[DSL Guide](dsl.md)**: Domain Specific Language for automation

## Quick Start

### Docker (Recommended)

```bash
# Clone and start
git clone https://github.com/wronai/intent.git
cd intent
cp .env.example .env
# Edit .env with your settings

# Start all services
docker-compose up -d

# Open browser
open http://localhost/examples/usecases/
```

### Manual Installation

```bash
pip install -e ".[server]"
python examples/server.py
```

## Examples

| Example | Description | URL |
|---------|-------------|-----|
| Chat Assistant | AI chat with LLM | `/examples/usecases/05_chat_assistant.html` |
| File Upload | Vision + OCR + Document extraction | `/examples/usecases/08_file_upload.html` |
| Smart Home | Voice command processing | `/examples/usecases/07_iot_smart_home.html` |
| Analytics | Dashboard with NLP queries | `/examples/usecases/06_dashboard_analytics.html` |

## CLI Commands

```bash
# List services
intentforge services

# Call service directly
intentforge dsl-call chat send '{"message": "Hello!"}'

# Run DSL script
intentforge dsl -f examples/dsl/01_chat_example.dsl

# Interactive REPL
intentforge repl
```

## API Endpoints (Code Execution)

| Endpoint | Description |
|----------|-------------|
| `POST /api/code/execute` | Simple code execution |
| `POST /api/code/autofix` | Execute with auto-fix loop |
| `POST /api/code/test-and-fix` | TDD with auto-fix |
| `POST /api/code/auto-conversation` | Branching conversations |
| `POST /api/code/git-tracked` | Git iteration tracking |

## Development

```bash
# Setup
pip install -e ".[dev]"
pre-commit install

# Test
pytest tests/unit/ -v --cov=intentforge

# Lint
ruff check intentforge/
ruff format intentforge/
```

See [TODO.md](../TODO.md) for roadmap and contribution guide.

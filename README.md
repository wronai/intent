# IntentForge 🚀

**Intent-Driven Development Framework**

IntentForge turns your natural language descriptions into executable code, dynamically and safely. It now features a powerful **Frontend Auto-Handler** that makes your HTML elements "smart" without writing custom JavaScript.

> **[Read the Full Documentation](docs/INDEX.md)**

## Key Features

- **🤖 LLM-Powered**: Uses local (Ollama) or cloud (Anthropic, OpenAI) models.
- **⚡ Dynamic Generation**: Generates, validates, and fixes code on the fly.
- **🛡️ Secure Sandbox**: Executes code in a restricted environment.
- **🌐 Frontend Auto-Handler**: Declarative `intent="..."` attributes for forms and tables.

## Quick Start (Frontend Example)

1. **Install**:
   ```bash
   pip install -r requirements.txt
   pip install fastapi uvicorn python-multipart
   ```

2. **Run Server**:
   ```bash
   python3 examples/server.py
   ```

3. **Visit**: `http://localhost:8085/form.html`

## Documentation

- **[Setup Guide](docs/setup.md)**
- **[Usage Examples](docs/usage.md)**
- **[Architecture](docs/architecture.md)**
- **[API Reference](docs/api.md)**

## License

MIT

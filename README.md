# IntentForge

> **NLP-driven Code Generation Framework** - Generate backend, frontend, and firmware code from natural language via universal MQTT protocol.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

IntentForge transforms natural language descriptions into production-ready code. It works across any software layer - from static HTML pages to embedded firmware - using MQTT as a universal communication protocol.

```
"Stwórz endpoint do pobierania użytkowników z paginacją"
                    ↓
           [IntentForge]
                    ↓
    Complete FastAPI endpoint with validation,
    pagination, error handling, and tests
```

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **3-Level Validation** | Syntax → Security → Semantic validation before first use |
| **Smart Caching** | Fingerprint-based cache with TTL - reuses identical intents |
| **DSL Generators** | Native generators for SQL, DOM, API without LLM calls |
| **Universal Protocol** | MQTT enables any client (HTML, CLI, mobile, IoT) |
| **Secure by Default** | Sandbox execution, injection detection, .env secrets |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Static   │  │ React    │  │ CLI      │  │ Mobile   │  │ ESP32    │      │
│  │ HTML/JS  │  │ WebGUI   │  │ Shell    │  │ App      │  │ Firmware │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
│       └─────────────┴─────────────┴─────────────┴─────────────┘             │
│                                   │                                         │
│                    Natural Language Intent (MQTT)                           │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MQTT BROKER                                          │
│  intentforge/intent/request/{client_id}  →  Intent requests                 │
│  intentforge/intent/response/{client_id} →  Generated code                  │
│  intentforge/capabilities                →  Server capabilities             │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      INTENTFORGE ENGINE                                      │
│                                                                              │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                  │
│  │   CACHE     │      │  VALIDATOR  │      │  GENERATOR  │                  │
│  │             │      │             │      │             │                  │
│  │ Fingerprint │      │ 1. Syntax   │      │ DSL:        │                  │
│  │ TTL         │      │ 2. Security │      │ - SQL       │                  │
│  │ Redis/Mem   │      │ 3. Semantic │      │ - DOM       │                  │
│  └─────────────┘      └─────────────┘      │ - API       │                  │
│                                            │ LLM Fallback│                  │
│                                            └─────────────┘                  │
│                                                                              │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                  │
│  │   CONFIG    │      │   SANDBOX   │      │  EXECUTOR   │                  │
│  │             │      │             │      │             │                  │
│  │ .env        │      │ Restricted  │      │ Test        │                  │
│  │ Pydantic    │      │ builtins    │      │ Deploy      │                  │
│  │ Secrets     │      │ No os/sys   │      │ Generate    │                  │
│  └─────────────┘      └─────────────┘      └─────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/softreck/intentforge.git
cd intentforge

# Install with all dependencies
make dev

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings (especially LLM_API_KEY)
```

### Basic Usage

```python
import asyncio
from intentforge import IntentForge, Intent, IntentType, TargetPlatform

async def main():
    # Initialize
    forge = IntentForge(
        api_key="sk-ant-...",  # Or set LLM_API_KEY in .env
        enable_auto_deploy=False,
        sandbox_mode=True
    )
    
    # Create intent
    intent = Intent(
        description="Create API endpoint to list users with pagination",
        intent_type=IntentType.API_ENDPOINT,
        target_platform=TargetPlatform.PYTHON_FASTAPI,
        context={
            "endpoint": "/api/users",
            "method": "GET",
            "model": "User"
        }
    )
    
    # Generate code
    result = await forge.process_intent(intent)
    
    if result.success:
        print(f"Generated {result.language} code:")
        print(result.generated_code)
        print(f"\nValidation passed: {result.validation_passed}")
    else:
        print(f"Errors: {result.validation_errors}")

asyncio.run(main())
```

### From Static HTML (via MQTT)

```html
<!DOCTYPE html>
<html>
<head>
    <script src="https://unpkg.com/mqtt/dist/mqtt.min.js"></script>
    <script src="intentforge-client.js"></script>
</head>
<body>
    <textarea id="intent" placeholder="Describe what you need..."></textarea>
    <button onclick="generateCode()">Generate</button>
    <pre id="output"></pre>
    
    <script>
        const client = new IntentForgeClient('ws://localhost:9001');
        
        async function generateCode() {
            const description = document.getElementById('intent').value;
            
            try {
                const result = await client.submitIntent({
                    description: description,
                    intent_type: 'api_endpoint',
                    target_platform: 'python_fastapi'
                });
                
                document.getElementById('output').textContent = result.generated_code;
            } catch (error) {
                console.error(error);
            }
        }
    </script>
</body>
</html>
```

### With Docker Compose

```bash
# Start full stack (server, MQTT, PostgreSQL, Redis)
docker-compose up -d

# View logs
docker-compose logs -f intentforge

# Stop
docker-compose down
```

## 📋 Validation Pipeline

### Level 1: Syntax Validation
- AST parsing for Python
- Bracket matching for JavaScript
- SQL keyword validation
- HTML tag structure

### Level 2: Security Validation
- Dangerous function detection (`eval`, `exec`, `os.system`)
- SQL injection patterns
- XSS vulnerabilities in DOM code
- Import restrictions

### Level 3: Semantic Validation
- Undefined variable detection
- Unreachable code analysis
- Type inference (basic)
- Cyclomatic complexity

```python
from intentforge import CodeValidator, ValidationLevel

validator = CodeValidator(sandbox_mode=True)

# Validate generated code
is_valid, errors = await validator.validate(
    code=generated_code,
    language="python",
    level=ValidationLevel.FULL
)

if not is_valid:
    print("Security issues found:", errors)
```

## 🔄 Caching System

IntentForge uses fingerprint-based caching to avoid regenerating identical code:

```python
# Fingerprint is generated from:
# - Description (normalized)
# - Intent type
# - Target platform
# - Context (optional)
# - Constraints (optional)

# Same intent = Cache hit (instant response)
intent1 = Intent(description="List users with pagination")
intent2 = Intent(description="List users with pagination")  # Cache hit!

# Different context = Different fingerprint
intent3 = Intent(
    description="List users with pagination",
    context={"auth_required": True}  # New fingerprint
)
```

Cache backends:
- **Memory** (default): Fast, limited by RAM
- **Redis**: Distributed, persistent
- **File**: Simple persistence

## 🔧 DSL Generators

Built-in generators for common patterns (no LLM required):

### SQL Generator
```python
intent = Intent(
    description="Pobierz wszystkich użytkowników z paginacją",
    intent_type=IntentType.DATABASE_SCHEMA,
    context={
        "table": "users",
        "columns": ["id", "name", "email"]
    }
)
# Generates parameterized SQL query
```

### DOM/Form Handler
```python
intent = Intent(
    description="Handle form submission and send to API",
    intent_type=IntentType.EVENT_HANDLER,
    context={
        "form_id": "contactForm",
        "endpoint": "/api/contact",
        "fields": ["name", "email", "message"]
    }
)
# Generates complete JavaScript form handler
```

### API Endpoint
```python
intent = Intent(
    description="CRUD endpoints for products",
    intent_type=IntentType.API_ENDPOINT,
    target_platform=TargetPlatform.PYTHON_FASTAPI,
    context={
        "endpoint": "/api/products",
        "model": "Product"
    }
)
# Generates FastAPI router with Pydantic models
```

## 📁 Project Structure

```
intentforge/
├── intentforge/
│   ├── __init__.py          # Package exports
│   ├── core.py              # Intent, IntentResult, IntentForge
│   ├── config.py            # Pydantic settings, .env loading
│   ├── broker.py            # MQTT communication
│   ├── generator.py         # Code generation (DSL + LLM)
│   ├── validator.py         # 3-level validation
│   ├── cache.py             # Fingerprinting and caching
│   └── executor.py          # Sandbox execution
├── tests/
│   ├── test_validator.py
│   ├── test_generator.py
│   └── test_cache.py
├── static/                  # Web UI files
├── config/
│   ├── mosquitto.conf       # MQTT broker config
│   └── nginx.conf           # Web server config
├── .env.example             # Environment template
├── docker-compose.yml       # Full stack deployment
├── Makefile                 # Build automation
├── pyproject.toml           # Package configuration
└── README.md
```

## 🛡️ Security

| Concern | Solution |
|---------|----------|
| **Credentials** | `.env` file, never hardcoded |
| **SQL Injection** | Parameterized queries enforced |
| **Code Execution** | Sandbox with restricted builtins |
| **XSS** | DOM code validation |
| **Dangerous Imports** | Blocked: `os`, `subprocess`, `ctypes` |

## 📊 Comparison

| Aspect | IntentForge | GitHub Copilot | ChatGPT |
|--------|-------------|----------------|---------|
| **Protocol** | MQTT (universal) | IDE plugin | Web/API |
| **Validation** | 3-level automatic | None | None |
| **Caching** | Fingerprint-based | None | None |
| **Embedded/IoT** | ✅ Native | ❌ | ❌ |
| **Offline** | ✅ DSL generators | ❌ | ❌ |
| **Self-hosted** | ✅ | ❌ | ❌ |

## 🔮 Roadmap

- [ ] Visual Studio Code extension
- [ ] Schema inference from database
- [ ] Multi-file project generation
- [ ] Custom DSL plugins
- [ ] Kubernetes deployment
- [ ] OpenAPI spec generation

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Run tests (`make test`)
4. Run linters (`make lint`)
5. Commit changes (`git commit -am 'Add amazing feature'`)
6. Push branch (`git push origin feature/amazing`)
7. Open Pull Request

---

**Made with ❤️ by [Softreck](https://softreck.dev)**

# IntentForge Architecture

> **Version**: 0.3.0-dev
> **Last Updated**: December 2024
> **Test Coverage**: 12% (20 tests passing)

## Project Structure (Post-Cleanup)

```
intentforge/
├── intentforge/              # Core library
│   ├── __init__.py          # Public API exports
│   ├── core.py              # IntentForge main class
│   ├── llm/                 # LLM providers
│   │   └── providers.py     # Ollama, Anthropic, OpenAI, LiteLLM
│   ├── dsl.py               # DSL lexer, parser, interpreter
│   ├── services.py          # Chat, File, Analytics, Vision services
│   ├── code_runner.py       # Auto-fix code execution
│   ├── code_tester.py       # TDD with auto-fix
│   ├── conversation_engine.py # Branching conversations
│   ├── git_manager.py       # Git iteration tracking
│   ├── modules.py           # Autonomous module manager
│   ├── autonomous.py        # Multi-step agent workflows
│   └── ...
├── examples/
│   ├── server.py            # FastAPI server
│   ├── usecases/            # HTML demo apps
│   ├── dsl/                 # DSL examples
│   └── python/              # Python examples
├── frontend/
│   └── sdk/                 # JavaScript SDK v2.0
├── modules/
│   └── .template/           # Module template
├── tests/
│   └── unit/                # Unit tests
├── docs/                    # Documentation
├── docker/                  # Docker configs
└── scripts/                 # Utility scripts
```

## Multi-Layer Modular Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        UI LAYER                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Chat UI     │  │ Code Editor │  │ Dashboard   │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API LAYER (FastAPI)                        │
│  /api/chat  /api/code/*  /api/modules/*  /api/autonomous/*      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CONVERSATION ENGINE                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              ThreadManager                              │    │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐            │    │
│  │  │ Thread 1  │  │ Thread 2  │  │ Thread N  │            │    │
│  │  │ (Main)    │──│ (Branch)  │──│ (Branch)  │            │    │
│  │  └───────────┘  └───────────┘  └───────────┘            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           ConversationBrancher                          │    │
│  │  • Detect problems                                      │    │
│  │  • Spawn sub-conversations                              │    │
│  │  • Merge results                                        │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LLM LAYER                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              LLMAnalyzer (replaces hardcoded logic)     │    │
│  │  • analyze_error() - Classify errors via LLM            │    │
│  │  • generate_fix() - Generate fixes via LLM              │    │
│  │  • should_branch() - Determine parallelization          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              ChatService                                │    │
│  │  • Ollama / OpenAI / Custom LLM                         │    │
│  │  • Streaming support                                    │    │
│  │  • Response caching                                     │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   EXECUTION LAYER                               │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │
│  │ CodeRunner    │  │ CodeTester    │  │ ModuleManager │        │
│  │ • Execute     │  │ • Generate    │  │ • Create      │        │
│  │ • Auto-fix    │  │   tests       │  │ • Build       │        │
│  │ • Install     │  │ • TDD loop    │  │ • Execute     │        │
│  │   deps        │  │ • Sandbox     │  │ • Manage      │        │
│  └───────────────┘  └───────────────┘  └───────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MODULE LAYER                                  │
│  modules/                                                       │
│  ├── .template/          # Base template for new modules        │
│  ├── csv_parser/         # Auto-generated module                │
│  ├── json_validator/     # Auto-generated module                │
│  └── [module_name]/      # Each module is a complete service    │
│      ├── Dockerfile                                             │
│      ├── main.py                                                │
│      ├── requirements.txt                                       │
│      └── module.yaml                                            │
└─────────────────────────────────────────────────────────────────┘
```

## Conversation Branching Flow

```
Main Conversation
       │
       ▼
  [Execute Code]
       │
       ▼
  [Error Detected?]──No──► [Success]
       │
      Yes
       │
       ▼
  [LLM Analyzes Error]
       │
       ▼
  [Problem Classification]
       │
       ├─► Problem 1: Missing Dependency
       │        │
       │        ▼
       │   [Branch Thread 1]
       │        │
       │        ▼
       │   [LLM: Generate Fix]
       │        │
       │        ▼
       │   [Execute & Verify]
       │        │
       │        ▼
       │   [Resolved? Loop if not]
       │
       ├─► Problem 2: Syntax Error
       │        │
       │        ▼
       │   [Branch Thread 2]
       │        │
       │        ▼
       │   [LLM: Generate Fix]
       │        │
       │        ▼
       │   [Execute & Verify]
       │
       └─► Problem N: ...
                │
                ▼
          [Parallel Resolution]
                │
                ▼
          [Merge Results]
                │
                ▼
          [Final Code]
```

## Key Components

### 1. ConversationEngine (`intentforge/conversation_engine.py`)

**ThreadManager**
- Coordinates parallel conversation threads
- Max parallel threads configurable
- Tracks active tasks

**ConversationBrancher**
- Detects problems using LLM (not hardcoded regex)
- Spawns sub-conversations for each problem
- Merges results back to parent

**LLMAnalyzer**
- Replaces hardcoded error patterns
- Uses LLM for error classification
- Generates fixes dynamically

### 2. CodeRunner (`intentforge/code_runner.py`)

- Execute code with auto-fix
- Install missing dependencies
- Retry loop until success

### 3. CodeTester (`intentforge/code_tester.py`)

- Generate tests from intent
- TDD fix loop
- Sandbox environment

### 4. ModuleManager (`intentforge/modules.py`)

- Create reusable service modules
- Build Docker containers
- Execute module actions

## API Endpoints

### Code Execution

| Endpoint | Description |
|----------|-------------|
| `POST /api/code/auto-conversation` | Execute with branching conversations |
| `POST /api/code/autofix` | Execute with auto-fix loop |
| `POST /api/code/test-and-fix` | TDD with auto-fix |
| `POST /api/code/execute` | Simple execution |

### Modules

| Endpoint | Description |
|----------|-------------|
| `GET /api/modules` | List modules |
| `POST /api/modules/create` | Create from code |
| `POST /api/modules/create-from-task` | Create from LLM |
| `POST /api/modules/{name}/execute` | Execute module |

### Autonomous

| Endpoint | Description |
|----------|-------------|
| `POST /api/autonomous/execute` | Multi-step LLM task |
| `GET /api/autonomous/history` | Task history |

## Configuration

### Environment Variables

```bash
# LLM Provider
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
OLLAMA_HOST=http://ollama:11434

# Execution
CODE_TIMEOUT=30
MAX_RETRIES=3
AUTO_INSTALL=true

# Conversation Engine
MAX_PARALLEL_THREADS=5
MAX_BRANCH_DEPTH=3
```

## LLM-Driven vs Hardcoded

### Before (Hardcoded)
```python
# Hardcoded regex patterns
if re.match(r"ModuleNotFoundError", error):
    return ErrorType.MISSING_MODULE
elif re.match(r"SyntaxError", error):
    return ErrorType.SYNTAX_ERROR
```

### After (LLM-Driven)
```python
# LLM analyzes dynamically
analysis = await llm_analyzer.analyze_error(error, code, context)
return ProblemType(analysis["error_type"])
```

### 5. GitManager (`intentforge/git_manager.py`)

Git-based iteration tracking for code changes:

```
fix/20241229_141500 (branch)
     │
     ├── Commit 1: Initial code
     ├── Commit 2: Attempt 1 - Fixed import
     └── Commit 3: Success - All tests passed
```

**Features:**
- Track each fix attempt as a commit
- Analyze diffs between iterations
- Provide Git history context to LLM
- Pattern analysis from previous fixes

### 6. LLM Providers (`intentforge/llm/providers.py`)

| Provider | Status | Notes |
|----------|--------|-------|
| Ollama | ✅ Default | Local, free |
| Anthropic | ✅ | Requires API key |
| OpenAI | ✅ | Requires API key |
| LiteLLM | ✅ | Universal adapter |

## Additional API Endpoints

### Git-Tracked Execution

| Endpoint | Description |
|----------|-------------|
| `POST /api/code/git-tracked` | Execute with Git iteration history |

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run tests with coverage
pytest tests/unit/ -v --cov=intentforge

# Current: 20 tests, 12% coverage
```

### Linting

```bash
# Check
ruff check intentforge/

# Format
ruff format intentforge/

# Current: 403 warnings (mostly style, not bugs)
```

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install

# Hooks: trim-whitespace, yaml-check, ruff, ruff-format
```

## Benefits

1. **No hardcoded patterns** - LLM adapts to any error type
2. **Parallel resolution** - Multiple problems fixed simultaneously
3. **Conversation tracking** - Full history of fix attempts
4. **Modular architecture** - Easy to extend and maintain
5. **Self-healing** - Continuous improvement through learning
6. **Git-based learning** - History context improves fixes
7. **Multi-provider LLM** - Ollama, Anthropic, OpenAI, LiteLLM

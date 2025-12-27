# API Reference

**[🏠 Home](INDEX.md) | [⚙️ Setup](setup.md) | [🛠️ Usage](usage.md) | [🏗️ Architecture](architecture.md) | [📚 API](api.md)**

This section documents the internal Python API for IntentForge.

## Core

### `IntentForge`
Main entry point for the application.

```python
from intentforge import IntentForge

forge = IntentForge(
    provider="ollama",
    model="llama3.1:8b",
    sandbox_mode=True
)
```

- **`process_intent(intent: Intent) -> IntentResult`**: Process an intent and return the result.

### `Intent`
Data class representing a user request.

- `description` (str): Natural language description.
- `intent_type` (IntentType): Type of intent (WORKFLOW, API_ENDPOINT, etc.).
- `context` (dict): Additional data (e.g., form fields).

## Generators

### `CodeGenerator` ([Source](../intentforge/generator.py))
Handles the heavy lifting of code generation.

- **`generate(intent: Intent) -> str`**:  Orchestrates the generation process with retries.

## Executor

### `DynamicExecutor` ([Source](../intentforge/executor.py))
Executes code in a controlled environment.

- **`execute(code: str, language: str, ...) -> ExecutionResult`**: Runs the code and returns output/return value.

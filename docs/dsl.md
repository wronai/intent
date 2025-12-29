# IntentForge DSL Guide

**[🏠 Home](INDEX.md) | [⚙️ Setup](setup.md) | [🛠️ Usage](usage.md) | [🏗️ Architecture](architecture.md) | [📚 API](api.md) | [🔧 Services](services.md) | [📝 DSL](dsl.md)**

IntentForge DSL (Domain Specific Language) provides a simple, human-readable syntax for defining and executing workflows.

## Quick Start

```bash
# Run DSL command
intentforge dsl -c 'chat.send(message="Hello!")'

# Run DSL file
intentforge dsl -f examples/dsl/01_chat_example.dsl

# Interactive REPL
intentforge repl

# Call service directly
intentforge dsl-call chat send '{"message": "Hello!"}'
```

## Syntax

### Service Calls

```dsl
service.action(param1="value", param2=123)
```

**Examples:**
```dsl
chat.send(message="Cześć!")
chat.models()
analytics.stats(period="today")
voice.process(command="Włącz światło")
file.ocr(image_base64="...")
```

### Variables

```dsl
$result = chat.send(message="Hello")
$response = $result.response
$models = chat.models()
```

### String Concatenation

```dsl
$name = "Jan"
$greeting = "Hello, " + $name + "!"
```

### Conditionals

```dsl
if $result.success then
    chat.send(message="OK")
else
    chat.send(message="Error")
end
```

### Loops

```dsl
for item in $items do
    chat.send(message=$item.name)
end
```

### Pipelines

```dsl
file.analyze(filename="img.jpg") | chat.send(message=$result.description)
```

### Comments

```dsl
# This is a comment
$result = chat.send(message="Hello")  # Inline comment
```

## CLI Commands

### `intentforge services`

List all available services:

```bash
$ intentforge services
Available services:
  email: Email service using SMTP configuration from .env
  payment: Payment processing with PayPal, Stripe, P24
  form: Form handling service
  camera: Camera and image analysis service
  data: Generic data operations service
  chat: Chat service using LLM from .env configuration
  analytics: Analytics service using LLM for data generation
  voice: Voice command processing using LLM for NLP
  file: File processing service with Vision and OCR
```

### `intentforge services <name>`

Show service details:

```bash
$ intentforge services chat
Service: chat
Class: ChatService

Actions:
  models()
    List available models (for Ollama)
  send(message, model=None, history=None, system=None, stream=False)
    Send message to LLM and get response
```

### `intentforge dsl`

Run DSL script:

```bash
# From file
intentforge dsl -f script.dsl

# From command line
intentforge dsl -c 'chat.send(message="Hello")'

# From stdin
echo 'analytics.stats(period="today")' | intentforge dsl
```

### `intentforge dsl-call`

Call service action directly:

```bash
intentforge dsl-call <service> <action> '<json_args>'

# Examples
intentforge dsl-call chat send '{"message": "Hello!"}'
intentforge dsl-call analytics stats '{"period": "today"}'
intentforge dsl-call voice process '{"command": "Turn on lights"}'
intentforge dsl-call file ocr '{"image_base64": "..."}'
```

### `intentforge dsl-gen`

Generate code from DSL:

```bash
# Generate Python
intentforge dsl-gen -f script.dsl -t python > script.py

# Generate Shell
intentforge dsl-gen -f script.dsl -t shell > script.sh
```

### `intentforge repl`

Interactive REPL:

```
$ intentforge repl
IntentForge DSL REPL
Type 'help' for commands, 'exit' to quit

dsl> services
  chat
  analytics
  voice
  file
  ...

dsl> chat.models()
{
  "success": true,
  "models": ["llama3.1:8b", "llava:13b", ...],
  "provider": "ollama"
}

dsl> $result = chat.send(message="Cześć!")
dsl> $result.response
"Witaj! Jak mogę Ci pomóc?"

dsl> vars
  $result = {"success": true, "response": "Witaj!...", ...}

dsl> exit
Bye!
```

## Example Scripts

### Chat Example

```dsl
# examples/dsl/01_chat_example.dsl

# List available models
$models = chat.models()

# Send a message
$response = chat.send(message="Cześć! Jak się masz?")

# Print the response
$response
```

### Analytics Example

```dsl
# examples/dsl/02_analytics_example.dsl

# Get current stats
$stats = analytics.stats(period="current_month")

# Natural language query
$query_result = analytics.query(query="Pokaż sprzedaż z ostatniego tygodnia")

# Get top products
$products = analytics.products(limit=5)
```

### Voice Command Example

```dsl
# examples/dsl/03_voice_example.dsl

# Process voice command with LLM NLP
$cmd = voice.process(command="Włącz światło w salonie")

# Check result
if $cmd.success then
    $cmd.response
else
    "Nie rozpoznano komendy"
end
```

### Workflow Example

```dsl
# examples/dsl/05_workflow_example.dsl

# Step 1: Get analytics data
$stats = analytics.stats(period="today")

# Step 2: Generate report with chat
$report_prompt = "Wygeneruj raport: przychód=" + $stats.revenue
$report = chat.send(message=$report_prompt)

# Step 3: Process voice notification
$notification = voice.process(command="Wyślij powiadomienie")

# Return final report
$report
```

## Shell Integration

```bash
# One-liner commands
intentforge dsl -c 'chat.send(message="Hello")'

# Pipe DSL from stdin
echo 'analytics.stats(period="today")' | intentforge dsl

# Use in shell scripts
STATS=$(intentforge dsl-call analytics stats '{"period": "today"}')
echo $STATS | jq '.revenue'

# Generate and run Python
intentforge dsl-gen -f script.dsl -t python > script.py
python script.py
```

## Code Generation

DSL can be converted to Python or Shell scripts:

### Generated Python

```python
#!/usr/bin/env python3
"""Auto-generated from IntentForge DSL"""

import asyncio
from intentforge.services import services

async def main():
    result = await services.get("chat").send(message="Hello")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Generated Shell

```bash
#!/bin/bash
# Auto-generated from IntentForge DSL

set -e

intentforge dsl-call chat send '{"message": "Hello"}'
```

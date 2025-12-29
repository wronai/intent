# IntentForge DSL Examples

Domain Specific Language for IntentForge - reusable functions callable from shell.

## Quick Start

```bash
# List available services
intentforge services

# Show service details
intentforge services chat

# Run DSL command directly
intentforge dsl -c 'chat.models()'

# Run DSL file
intentforge dsl -f examples/dsl/01_chat_example.dsl

# Interactive REPL
intentforge repl

# Call service directly
intentforge dsl-call chat send '{"message": "Hello!"}'

# Generate Python code from DSL
intentforge dsl-gen -f examples/dsl/01_chat_example.dsl -t python

# Generate Shell script from DSL
intentforge dsl-gen -f examples/dsl/01_chat_example.dsl -t shell
```

## DSL Syntax

### Service Calls
```dsl
service.action(param1="value", param2=123)
```

### Variables
```dsl
$result = chat.send(message="Hello")
$response = $result.response
```

### String Concatenation
```dsl
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
```

## Available Services

| Service | Description |
|---------|-------------|
| `chat` | LLM chat (Ollama/Anthropic/OpenAI) |
| `analytics` | Analytics data and NLP queries |
| `voice` | Voice command processing (NLP) |
| `file` | File analysis and Vision (LLaVA) |
| `data` | Generic data operations |
| `form` | Form handling |
| `payment` | Payment processing |
| `email` | Email sending |
| `camera` | Camera/video operations |

## Examples

### Chat
```dsl
# List models
$models = chat.models()

# Send message
$response = chat.send(message="Cześć!")
```

### Analytics
```dsl
# Get stats
$stats = analytics.stats(period="current_month")

# NLP query
$result = analytics.query(query="Pokaż sprzedaż z ostatniego tygodnia")
```

### Voice Commands
```dsl
# Process voice command with LLM
$cmd = voice.process(command="Włącz światło w salonie")
```

### File & Vision
```dsl
# Analyze text file
$analysis = file.analyze(filename="data.csv", content="...", options={"analyze": true})

# OCR from image (requires base64)
$ocr = file.ocr(image_base64="...")

# Describe image
$desc = file.describe(image_base64="...")

# Detect objects
$objects = file.detect_objects(image_base64="...")
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

## REPL Mode

```
$ intentforge repl
IntentForge DSL REPL
Type 'help' for commands, 'exit' to quit

dsl> services
  chat
  analytics
  voice
  file
  data
  form
  payment
  email
  camera

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

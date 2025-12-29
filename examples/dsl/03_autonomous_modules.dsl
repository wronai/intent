# IntentForge DSL - Autonomous Modules Example
# Demonstrates creating and using reusable service modules

# List available modules
$modules = module.list()

# Create a new module from natural language task
$parser_result = module.create_from_task(
    task="Create a JSON validator that checks if input is valid JSON and returns parsed data or error",
    name="json_validator"
)

# Create module with custom code
$custom_module = module.create(
    name="text_processor",
    description="Text processing utilities",
    code="def process(data):\n    text = data.get('text', '')\n    return {'word_count': len(text.split()), 'char_count': len(text)}"
)

# Use the autonomous agent for complex tasks
$agent_result = agent.execute(
    task="Analyze the structure of a Python project and create a summary",
    context={"path": "/app"}
)

# Chain multiple LLM calls proactively
$analysis = chat.send(message="What are the key features of microservices architecture?")

# Use the response in next call
$implementation = chat.send(
    message="Based on this analysis, suggest a Python implementation: " + $analysis.response
)

# Output results
$modules
$parser_result
$agent_result

# IntentForge DSL - Chat Example
# Run: intentforge dsl -f examples/dsl/01_chat_example.dsl

# List available models
$models = chat.models()

# Send a message to the chat
$response = chat.send(message="Cześć! Jak się masz?")

# Print the response
$response

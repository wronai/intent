# IntentForge DSL - Voice Command Example
# Run: intentforge dsl -f examples/dsl/03_voice_example.dsl

# Process voice command with LLM NLP
$cmd = voice.process(command="Włącz światło w salonie")

# Check result
if $cmd.success then
    $cmd.response
else
    "Nie rozpoznano komendy"

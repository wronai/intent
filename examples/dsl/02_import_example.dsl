# IntentForge DSL - Import Example
# Demonstrates importing functions from other DSL files

# Import utils module
import "utils.dsl" as utils

# Use imported function
$greeting = utils.greet("Świat")

# Print version from imported module
$version = utils.VERSION

# Local function
func say_hello() do
    return chat.send(message="Hello from local function!")
end

# Call local function
$local_result = say_hello()

# Output results
$greeting
$version

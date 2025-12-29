# IntentForge DSL - Utility Functions
# This file can be imported by other DSL scripts

# Helper function to greet
func greet(name) do
    $message = "Cześć, " + name + "!"
    return chat.send(message=$message)
end

# Helper function to analyze and describe
func analyze_image(image_data) do
    $analysis = file.analyze(image_base64=image_data)
    return $analysis
end

# Export functions for use in other modules
export greet
export analyze_image

# Exported variable
$VERSION = "1.0.0"
export VERSION

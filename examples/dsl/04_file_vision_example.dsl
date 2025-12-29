# IntentForge DSL - File & Vision Example
# Run: intentforge dsl -f examples/dsl/04_file_vision_example.dsl

# Analyze text file
$text_analysis = file.analyze(filename="data.csv", content="id,name,value\n1,test,100", options={"analyze": true})

# Extract structured data
$extracted = file.extract(filename="data.json", content='{"name": "John", "age": 30}')

# Vision - describe image (requires base64 image data)
# $description = file.describe(image_base64="...")

# Vision - OCR from image
# $ocr = file.ocr(image_base64="...")

# Vision - detect objects
# $objects = file.detect_objects(image_base64="...")

$text_analysis

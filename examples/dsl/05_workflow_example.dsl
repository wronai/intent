# IntentForge DSL - Complete Workflow Example
# Run: intentforge dsl -f examples/dsl/05_workflow_example.dsl

# This example shows a complete workflow combining multiple services

# Step 1: Get analytics data
$stats = analytics.stats(period="today")

# Step 2: Generate report with chat
$report_prompt = "Wygeneruj krótki raport na podstawie danych: przychód=" + $stats.revenue + ", zamówienia=" + $stats.orders
$report = chat.send(message=$report_prompt)

# Step 3: Process voice command for notification
$notification = voice.process(command="Wyślij powiadomienie o raporcie")

# Return final report
$report

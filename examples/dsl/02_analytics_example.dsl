# IntentForge DSL - Analytics Example
# Run: intentforge dsl -f examples/dsl/02_analytics_example.dsl

# Get current stats
$stats = analytics.stats(period="current_month")

# Get chart data for revenue
$chart = analytics.chart_data(metric="revenue", period="week")

# Natural language query
$query_result = analytics.query(query="Pokaż sprzedaż z ostatniego tygodnia")

# Get top products
$products = analytics.products(limit=5)

# Print results
$stats

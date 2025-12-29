# IntentForge Examples

This directory contains practical examples demonstrating IntentForge capabilities.

## 📁 Structure

```
examples/
├── python/                    # Python examples
│   ├── ollama_example.py      # Direct Ollama integration
│   ├── api_client_example.py  # REST API client usage
│   └── mqtt_realtime_example.py # MQTT real-time communication
│
├── usecases/                  # Complete use case demos (HTML)
│   ├── 01_contact_form.html   # Contact form with email
│   ├── 02_ebook_payment.html  # E-commerce payment flow
│   ├── 03_camera_monitoring.html # AI camera monitoring
│   ├── 04_todo_app.html       # Todo application with CRUD
│   ├── 05_chat_assistant.html # AI chat assistant
│   ├── 06_dashboard_analytics.html # Analytics dashboard
│   ├── 07_iot_smart_home.html # Smart home IoT control
│   └── 08_file_upload.html    # File upload & AI processing
│
├── static/                    # Static assets
│   └── intent.js              # Simple declarative handler
│
├── dev_workflow.py            # Developer workflow with caching
├── example1_oneliner.py       # One-liner API examples
├── example2_static_html.html  # Static HTML + MQTT
└── example3_docker_workflow.py # Complete Docker workflow
```

## 🚀 Quick Start

### Prerequisites

```bash
# Start IntentForge services
make start

# Or manually
docker-compose up -d
```

### Python Examples

```bash
# Activate virtual environment
source venv/bin/activate

# Run Ollama example
python examples/python/ollama_example.py

# Run API client example
python examples/python/api_client_example.py

# Run MQTT example
python examples/python/mqtt_realtime_example.py
```

### HTML Examples

Open in browser after starting services:

- http://localhost/examples/usecases/01_contact_form.html
- http://localhost/examples/usecases/04_todo_app.html
- http://localhost/examples/usecases/05_chat_assistant.html

## 📚 Example Descriptions

### Python Examples

#### `ollama_example.py`
Direct integration with Ollama for local LLM inference:
- Code generation with CodeLlama
- Streaming responses
- Model listing and selection

#### `api_client_example.py`
Complete REST API client demonstrating:
- Health checks
- Code generation
- Intent processing
- CRUD data operations
- Form submission

#### `mqtt_realtime_example.py`
Real-time MQTT communication:
- Publish/Subscribe patterns
- Intent processing via MQTT
- IoT device simulation
- Sensor data streaming
- Event-driven architecture

### Use Case Examples

#### 01. Contact Form
Simple contact form with:
- Automatic form binding
- Email sending
- Success/error handling

#### 02. E-book Payment
E-commerce payment flow:
- Product display
- Discount codes
- Multiple payment providers (PayPal, Stripe, P24)
- Order confirmation

#### 03. Camera Monitoring
AI-powered camera system:
- Real-time video feed
- Motion/person/vehicle detection
- Alert configuration
- Event logging

#### 04. Todo App
Full CRUD todo application:
- Add/edit/delete tasks
- Priority levels
- Due dates
- Filtering and stats
- Real-time sync

#### 05. Chat Assistant
AI chat interface:
- Conversation history
- Code highlighting
- Quick action buttons
- Model selection

#### 06. Dashboard Analytics
Analytics dashboard with:
- KPI cards
- Interactive charts
- Natural language queries
- Real-time updates
- Data export

#### 07. Smart Home IoT
Home automation control:
- Device control (lights, AC, blinds)
- Scene activation
- Sensor monitoring
- Energy tracking
- Voice commands
- Automation rules

#### 08. File Upload
File processing system:
- Drag & drop upload
- OCR text extraction
- AI content analysis
- Multiple file types
- Progress tracking

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
API_URL=http://localhost:8001
MQTT_HOST=localhost
MQTT_PORT=1883

# LLM Configuration
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
OLLAMA_HOST=http://localhost:11434
```

### SDK Configuration

```html
<!-- Auto-configuration via data attributes -->
<script src="/sdk/intentforge.js"
        data-broker="ws://localhost:9001"
        data-auto-bind="true"
        data-debug="true">
</script>
```

## 🎯 Intent Attributes

### Form Intent
```html
<form data-intent="contact" data-intent-reset="true">
    <input type="text" name="name" required>
    <button type="submit">Send</button>
</form>
```

### Table Intent (Auto-load data)
```html
<table intent="Return a list of 10 users with id, name, and email">
    <thead><tr><th>ID</th><th>Name</th><th>Email</th></tr></thead>
    <tbody></tbody>
</table>
```

### Button Intent
```html
<button intent="Generate a random password with 16 characters">
    Generate Password
</button>
```

## 📡 MQTT Topics

```
intentforge/
├── intent/
│   ├── request/{client_id}    # Send intents
│   └── response/{client_id}   # Receive responses
├── devices/
│   ├── {device_id}/command    # Device commands
│   └── {device_id}/state      # Device state updates
├── sensors/
│   └── {sensor_type}          # Sensor readings
└── events/
    └── {event_type}           # System events
```

## 🧪 Testing Examples

```bash
# Test API endpoint
curl -X POST http://localhost:8001/api/intent \
  -H "Content-Type: application/json" \
  -d '{"description": "Return hello world", "intent_type": "workflow"}'

# Test health
curl http://localhost:8001/health
```

## 📖 More Information

- [Main Documentation](../DOCUMENTATION.md)
- [API Reference](../docs/api.md)
- [Architecture](../docs/architecture.md)

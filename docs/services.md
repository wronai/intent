# IntentForge Services

**[🏠 Home](INDEX.md) | [⚙️ Setup](setup.md) | [🛠️ Usage](usage.md) | [🏗️ Architecture](architecture.md) | [📚 API](api.md) | [🔧 Services](services.md) | [📝 DSL](dsl.md)**

IntentForge provides a set of LLM-powered services accessible via REST API.

## Available Services

| Service | Description | Key Actions |
|---------|-------------|-------------|
| `chat` | LLM chat interface | `send`, `models` |
| `analytics` | Analytics and NLP queries | `stats`, `chart_data`, `query`, `products` |
| `voice` | Voice command processing | `process` |
| `file` | File analysis and Vision | `analyze`, `ocr`, `process_document`, `describe` |
| `data` | Generic data operations | `list`, `get`, `create`, `update`, `delete` |
| `form` | Form handling | `submit`, `validate` |
| `payment` | Payment processing | `create`, `verify` |
| `email` | Email sending | `send` |
| `camera` | Camera operations | `snapshot`, `analyze` |

## API Endpoint

All services are accessible via:

```
POST /api/{service}
Content-Type: application/json

{
  "action": "action_name",
  "param1": "value1",
  ...
}
```

---

## Chat Service

LLM-powered chat using Ollama, Anthropic, or OpenAI (configured in `.env`).

### Actions

#### `send` - Send message to LLM

```bash
curl -X POST http://localhost:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "action": "send",
    "message": "Cześć! Jak się masz?",
    "model": "llama3.1:8b",
    "history": [],
    "system": "Jesteś pomocnym asystentem."
  }'
```

**Response:**
```json
{
  "success": true,
  "response": "Witaj! Jestem dobrze, dziękuję!",
  "model": "llama3.1:8b",
  "provider": "ollama",
  "input_tokens": 73,
  "output_tokens": 44,
  "total_tokens": 117,
  "latency_ms": 1228.57
}
```

#### `models` - List available models

```bash
curl -X POST http://localhost:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"action": "models"}'
```

---

## Analytics Service

LLM-powered analytics data generation and natural language queries.

### Actions

#### `stats` - Get statistics

```bash
curl -X POST http://localhost:8001/api/analytics \
  -H "Content-Type: application/json" \
  -d '{"action": "stats", "period": "current_month"}'
```

**Response:**
```json
{
  "success": true,
  "revenue": 125430,
  "revenue_change": 12.5,
  "orders": 1847,
  "orders_change": 8.3,
  "users": 4521,
  "users_change": 15.2,
  "conversion": 3.2,
  "conversion_change": -0.5
}
```

#### `query` - Natural language query

```bash
curl -X POST http://localhost:8001/api/analytics \
  -H "Content-Type: application/json" \
  -d '{"action": "query", "query": "Pokaż sprzedaż z ostatniego tygodnia"}'
```

#### `chart_data` - Get chart data

```bash
curl -X POST http://localhost:8001/api/analytics \
  -H "Content-Type: application/json" \
  -d '{"action": "chart_data", "metric": "revenue", "period": "week"}'
```

#### `products` - Get top products

```bash
curl -X POST http://localhost:8001/api/analytics \
  -H "Content-Type: application/json" \
  -d '{"action": "products", "limit": 5}'
```

---

## Voice Service

NLP-powered voice command processing using LLM.

### Actions

#### `process` - Process voice command

```bash
curl -X POST http://localhost:8001/api/voice \
  -H "Content-Type: application/json" \
  -d '{"action": "process", "command": "Włącz światło w salonie", "language": "pl"}'
```

**Response:**
```json
{
  "success": true,
  "actions": [
    {
      "type": "device",
      "device_id": "living_light",
      "state": {"on": true}
    }
  ],
  "response": "Światło w salonie zostało włączone"
}
```

---

## File Service

File analysis with Vision AI (LLaVA) and Tesseract OCR.

### Actions

#### `analyze` - Analyze image with Vision

```bash
curl -X POST http://localhost:8001/api/file \
  -H "Content-Type: application/json" \
  -d '{
    "action": "analyze",
    "filename": "photo.jpg",
    "image_base64": "...",
    "file_type": "image/jpeg"
  }'
```

#### `ocr` - Extract text from image

Two-phase OCR: Tesseract (precise) → Vision (fallback)

```bash
curl -X POST http://localhost:8001/api/file \
  -H "Content-Type: application/json" \
  -d '{
    "action": "ocr",
    "image_base64": "...",
    "use_tesseract": true
  }'
```

**Response:**
```json
{
  "success": true,
  "text": "Extracted text from image...",
  "method": "tesseract",
  "model": "tesseract"
}
```

#### `process_document` - Full document processing pipeline

Three-phase processing:
1. **Vision** - Detect document type (receipt, invoice, ID, etc.)
2. **Tesseract OCR** - Precise text extraction
3. **LLM** - Structure extracted data

```bash
curl -X POST http://localhost:8001/api/file \
  -H "Content-Type: application/json" \
  -d '{
    "action": "process_document",
    "image_base64": "..."
  }'
```

**Response:**
```json
{
  "success": true,
  "document_type": "receipt",
  "phases": {
    "vision": {"success": true, "analysis": "...", "model": "llava:13b"},
    "ocr": {"success": true, "text": "...", "method": "tesseract", "char_count": 1234},
    "extraction": {"success": true, "structured_data": {...}}
  },
  "extracted_data": {
    "nazwa_sklepu": "Biedronka",
    "nip": "123-456-78-90",
    "produkty": [{"nazwa": "Mleko", "cena": "3.99"}],
    "suma": "45.99 PLN"
  },
  "raw_text": "..."
}
```

**Supported document types:**
- `receipt` - Paragony, rachunki kasowe
- `invoice` - Faktury VAT
- `id_card` - Dowody osobiste
- `drivers_license` - Prawa jazdy
- `bill` - Rachunki za usługi
- `contract` - Umowy

#### `describe` - Describe image content

```bash
curl -X POST http://localhost:8001/api/file \
  -H "Content-Type: application/json" \
  -d '{"action": "describe", "image_base64": "..."}'
```

#### `detect_objects` - Detect objects in image

```bash
curl -X POST http://localhost:8001/api/file \
  -H "Content-Type: application/json" \
  -d '{"action": "detect_objects", "image_base64": "..."}'
```

---

## Configuration

Services are configured via `.env`:

```bash
# LLM Provider
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
OLLAMA_HOST=http://localhost:11434

# Vision (for image analysis)
VISION_PROVIDER=ollama
VISION_MODEL=llava:13b
VISION_MAX_TOKENS=2048
```

## CLI Access

```bash
# List all services
intentforge services

# Show service details
intentforge services chat

# Call service directly
intentforge dsl-call chat send '{"message": "Hello!"}'
intentforge dsl-call analytics stats '{"period": "today"}'
intentforge dsl-call voice process '{"command": "Turn on lights"}'
```

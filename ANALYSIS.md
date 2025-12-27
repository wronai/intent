# 🔍 IntentForge - Analiza Projektu i Plan Refaktoryzacji

## ❌ Zidentyfikowane Problemy

### 1. Duplikaty Plików

| Plik | Lokalizacja 1 | Lokalizacja 2 | Status |
|------|---------------|---------------|--------|
| **JS SDK** | `/sdk/intentforge.js` (730 linii) | `/intentforge/static/js/intentforge-client.js` (364 linie) | ⚠️ Dwa różne API! |
| **ENV** | `.env.example` | `.env.complete.example` | ⚠️ Duplikat |
| **Static** | `/static/index.html` | `/intentforge/static/` | ⚠️ Dwa foldery |

### 2. Niespójna Struktura Folderów

```
❌ PRZED (aktualnie):
intentforge/
├── sdk/                      # SDK oddzielnie
├── static/                   # Frontend root
├── intentforge/
│   ├── static/               # ❌ DUPLIKAT
│   │   └── js/               # ❌ Inny JS client
│   └── ...
├── examples/
│   ├── example1_*.py
│   ├── example2_*.html       # ❌ Mieszanka
│   ├── usecases/             # ❌ Podkatalog
└── config/                   # Tylko mosquitto/nginx
```

### 3. Brak Wsparcia LLM

| Provider | Status | Rozwiązanie |
|----------|--------|-------------|
| Anthropic | ✅ | Wbudowane |
| OpenAI | ✅ | Wbudowane |
| **Ollama** | ❌ Brak | 🆕 Dodane w `llm/providers.py` |
| **LiteLLM** | ❌ Brak | 🆕 Dodane w `llm/providers.py` |

### 4. Bezpieczeństwo Frontend

| Problem | Ryzyko | Status |
|---------|--------|--------|
| Brak rate limiting | Wysokie | 🆕 Naprawione |
| Brak walidacji | Średnie | 🆕 Naprawione |
| Brak sanityzacji | Wysokie | 🆕 Naprawione |
| Brak CSRF | Średnie | 🆕 Naprawione |
| Brak offline queue | Niskie | 🆕 Naprawione |

---

## ✅ Wykonane Naprawy

### A. Nowy Moduł LLM z Ollama i LiteLLM

**Lokalizacja:** `/intentforge/llm/providers.py`

```python
# Użycie Ollama (lokalnie)
from intentforge.llm import get_llm_provider

llm = get_llm_provider("ollama", model="llama3")
response = await llm.generate("Create REST API for users")

# Użycie LiteLLM (dowolny backend)
llm = get_llm_provider("litellm", model="ollama/codellama")
response = await llm.generate_code("Create MQTT handler")

# Automatyczne wykrywanie z .env
llm = get_llm_provider()  # Czyta LLM_PROVIDER z .env
```

**Wspierane modele:**
- `anthropic` - Claude 3 Opus/Sonnet/Haiku
- `openai` - GPT-4o, GPT-4 Turbo
- `ollama` - llama3, codellama, mistral, phi3
- `litellm` - 100+ modeli przez jeden API

### B. Zunifikowane SDK JavaScript v2.0

**Lokalizacja:** `/frontend/sdk/intentforge.js`

**Nowe funkcje bezpieczeństwa:**

```javascript
// Rate limiting (60 req/min domyślnie)
const api = await IntentForge.init({
    enableRateLimit: true,
    rateLimitPerMinute: 60
});

// Walidacja przed wysłaniem
api.form('contact')
   .rules({
       email: { required: true, type: 'email' },
       message: { required: true, minLength: 10 }
   })
   .submit(data);

// Sanityzacja automatyczna (XSS protection)
// Wszystkie dane są automatycznie sanityzowane

// CSRF protection
// Token pobierany automatycznie z cookie/API

// Offline queue
const api = await IntentForge.init({
    enableOfflineQueue: true,
    maxQueueSize: 100
});
// Requesty są kolejkowane gdy offline i wysyłane po powrocie online
```

### C. Docker z Ollama i LiteLLM

**Lokalizacja:** `/docker/docker-compose.yml`

```bash
# Domyślnie (Anthropic API)
docker-compose up -d

# Z lokalnym Ollama
docker-compose --profile ollama up -d
docker exec intentforge-ollama ollama pull llama3

# Z LiteLLM (100+ modeli)
docker-compose --profile litellm up -d

# Wszystko razem
docker-compose --profile full up -d
```

---

## 📁 Nowa Struktura (Zalecana)

```
intentforge/
├── 📁 src/intentforge/              # Kod Python
│   ├── llm/                         # 🆕 Moduł LLM
│   │   ├── __init__.py
│   │   └── providers.py             # Ollama, LiteLLM, etc.
│   ├── services/                    # Serwisy
│   └── ...
│
├── 📁 frontend/                     # 🆕 Zunifikowany frontend
│   ├── sdk/
│   │   └── intentforge.js           # SDK v2.0 z security
│   └── index.html
│
├── 📁 examples/
│   ├── python/
│   └── html/
│
├── 📁 docker/                       # 🆕 Docker osobno
│   ├── docker-compose.yml           # Z Ollama/LiteLLM
│   ├── Dockerfile
│   └── config/
│       ├── mosquitto.conf
│       ├── nginx.conf
│       └── litellm_config.yaml      # 🆕 Konfiguracja LiteLLM
│
├── .env.example                     # Jeden plik (usunąć duplikat)
└── pyproject.toml
```

---

## 🔧 Konfiguracja .env dla Ollama/LiteLLM

```env
# =============================================================================
# LLM Provider Configuration
# =============================================================================

# Opcja 1: Anthropic (wymaga klucza API)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-xxx

# Opcja 2: Ollama (lokalnie, bez klucza)
LLM_PROVIDER=ollama
LLM_MODEL=llama3
OLLAMA_HOST=http://localhost:11434

# Opcja 3: LiteLLM (proxy dla wielu providerów)
LLM_PROVIDER=litellm
LLM_MODEL=ollama/codellama  # lub gpt-4o, claude-3-sonnet
LITELLM_API_BASE=http://localhost:4000

# Opcja 4: OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxx
LLM_MODEL=gpt-4o
```

---

## 🚀 Szybki Start z Ollama

```bash
# 1. Zainstaluj Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pobierz model
ollama pull llama3
ollama pull codellama  # dla generowania kodu

# 3. Skonfiguruj .env
echo "LLM_PROVIDER=ollama" >> .env
echo "LLM_MODEL=llama3" >> .env

# 4. Uruchom IntentForge
docker-compose --profile ollama up -d

# 5. Test
curl http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"description": "Create REST API for products"}'
```

---

## 📊 Podsumowanie Zmian

| Komponent | Przed | Po |
|-----------|-------|-----|
| JS SDK | 2 różne pliki | 1 zunifikowany |
| LLM Providers | 2 (Anthropic, OpenAI) | 4+ (+ Ollama, LiteLLM) |
| Bezpieczeństwo Frontend | Brak | Rate limiting, CSRF, Sanityzacja |
| Offline Support | Brak | Queue z localStorage |
| Docker Profiles | 1 | 4 (default, ollama, litellm, full) |
| Walidacja | Server-side tylko | Client + Server |

---

## ⚠️ Do Usunięcia (Duplikaty)

1. `/intentforge/static/js/intentforge-client.js` → Zastąpiony przez `/frontend/sdk/intentforge.js`
2. `/sdk/intentforge.js` → Przeniesiony do `/frontend/sdk/`
3. `.env.complete.example` → Połączyć z `.env.example`
4. `/static/` → Przenieść do `/frontend/`

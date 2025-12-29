# IntentForge - TODO / Roadmap

Lista planowanych ulepszeń i funkcji do wdrożenia.

---

## 🔴 KRYTYCZNE - Cleanup & Quality

### Duplikaty do usunięcia
- [ ] **Usuń duplikaty z root** - `intentforge.js`, `example*.py`, `simple.py`, `providers.py`, `ollama_example.py`
- [ ] **Usuń stare SDK** - `sdk/intentforge.js` → używaj `frontend/sdk/intentforge.js`
- [ ] **Usuń stary client** - `intentforge/static/js/intentforge-client.js`
- [ ] **Konsoliduj config** - `config/` → `docker/config/`
- [ ] **Merge docs** - `docs/architecture.md` + `ARCHITECTURE.md` → jeden plik
- [ ] **Przenieś DOCUMENTATION.md** → `docs/`

### Testy jednostkowe
- [ ] **test_llm_providers.py** - Testy dla Ollama, Anthropic, OpenAI, LiteLLM
- [ ] **test_code_runner.py** - Testy auto-fix, dependency install
- [ ] **test_conversation_engine.py** - Testy ThreadManager, ConversationBrancher
- [ ] **test_dsl.py** - Testy lexer, parser, interpreter
- [ ] **test_plugins.py** - Testy middleware, hooks
- [ ] **test_services.py** - Testy email, payment, camera
- [ ] **Cel: 80%+ coverage**

### CI/CD Pipeline
- [ ] **GitHub Actions** - `.github/workflows/ci.yml`
  - [ ] Lint (ruff)
  - [ ] Type check (mypy)
  - [ ] Unit tests
  - [ ] E2E tests
  - [ ] Build Docker image
- [ ] **Pre-commit hooks** - `.pre-commit-config.yaml`
- [ ] **Dependabot** - Automatyczne aktualizacje zależności

---

## 🔴 Wysokie priorytety

### Observability & Monitoring
- [ ] **Prometheus metrics** - `/metrics` endpoint
  - [ ] Request latency
  - [ ] LLM token usage
  - [ ] Error rates
  - [ ] Active conversations
- [ ] **OpenTelemetry tracing** - Distributed tracing
- [ ] **Structured logging** - JSON logs z correlation ID
- [ ] **Health checks** - `/health`, `/ready`, `/live`
- [ ] **Grafana dashboards** - Gotowe dashboardy

### Security Hardening
- [ ] **Input sanitization** - XSS, SQL injection prevention
- [ ] **Secrets management** - HashiCorp Vault / AWS Secrets Manager
- [ ] **CORS configuration** - Whitelist domen
- [ ] **Audit logging** - Logowanie wszystkich akcji
- [ ] **Rate limiting per endpoint** - Różne limity dla różnych endpointów

### Performance
- [ ] **Connection pooling** - PostgreSQL, Redis
- [ ] **Async everywhere** - Pełna asynchroniczność
- [ ] **Response compression** - gzip/brotli
- [ ] **CDN for static files** - CloudFlare/Fastly
- [ ] **Benchmarks** - Locust/k6 load tests

---

## 🟡 Średnie priorytety

### API Improvements
- [ ] **GraphQL API** - Alternatywa dla REST
- [ ] **API versioning** - `/api/v1/`, `/api/v2/`
- [ ] **OpenAPI docs** - Auto-generowana dokumentacja Swagger
- [ ] **Webhook system** - Outgoing webhooks dla integracji
- [ ] **Batch API** - Przetwarzanie wielu requestów

### Frontend Enhancements
- [ ] **Dark/Light mode** - Przełącznik motywu
- [ ] **PWA support** - Service worker, offline mode
- [ ] **Keyboard shortcuts** - Ctrl+Enter, etc.
- [ ] **Drag & drop files** - Upload przez przeciąganie
- [ ] **Code syntax highlighting** - Prism.js/highlight.js
- [ ] **Export chat** - Markdown/PDF/JSON

### New Web Components
- [ ] **`<intent-upload>`** - File upload z progress
- [ ] **`<intent-chat>`** - Wbudowany chat widget
- [ ] **`<intent-map>`** - Mapy z markerami
- [ ] **`<intent-calendar>`** - Kalendarz z eventami
- [ ] **`<intent-notification>`** - Push notifications

### CLI Improvements
- [ ] **Interactive setup** - `intentforge init`
- [ ] **Model management** - `intentforge model pull/list/remove`
- [ ] **Health check** - `intentforge doctor`
- [ ] **Logs viewer** - `intentforge logs -f`
- [ ] **REPL improvements** - Tab completion, history

---

## 🟢 Niskie priorytety / Nice-to-have

### Integrations
- [ ] **Slack bot** - `/intentforge ask ...`
- [ ] **Discord bot** - Bot dla serwerów Discord
- [ ] **VS Code extension** - IntelliSense dla DSL
- [ ] **Jupyter kernel** - DSL w Jupyter notebooks
- [ ] **Zapier/n8n** - Workflow automation
- [ ] **MCP server** - Model Context Protocol

### Advanced LLM Features
- [ ] **Multi-model routing** - Automatyczny wybór modelu
- [ ] **Prompt versioning** - Git-like wersjonowanie promptów
- [ ] **A/B testing** - Testowanie różnych promptów
- [ ] **Fine-tuning UI** - Interfejs do fine-tuningu
- [ ] **RAG integration** - Vector store (Chroma, Pinecone)
- [ ] **Agent memory** - Long-term memory dla agentów

### Enterprise Features
- [ ] **Multi-tenant** - Izolacja per organization
- [ ] **RBAC** - Role-based access control
- [ ] **SSO** - SAML/OIDC integration
- [ ] **Audit trail** - Compliance logging
- [ ] **Usage quotas** - Per-user/org limity
- [ ] **White-label** - Customizable branding

### Documentation
- [ ] **Video tutorials** - YouTube/Loom
- [ ] **Interactive playground** - Online sandbox
- [ ] **Cookbook** - Recipes dla common use cases
- [ ] **Architecture diagrams** - Mermaid/D2
- [ ] **Changelog** - CHANGELOG.md

---

## ✅ Ukończone

### Core Features
- [x] LLM integration (Ollama, Anthropic, OpenAI, LiteLLM)
- [x] Vision AI (LLaVA) - analiza obrazów
- [x] Tesseract OCR integration
- [x] Two-phase document processing pipeline
- [x] Chat service with history
- [x] Analytics service with NLP queries
- [x] Voice command processing

### DSL System
- [x] DSL with lexer, parser, interpreter
- [x] DSL import - `import "utils.dsl"`
- [x] DSL functions - `func name(params) do ... end`
- [x] DSL debugger - breakpoints, step-through
- [x] Streaming responses
- [x] Error recovery

### Code Execution
- [x] Code execution - zapisywanie i uruchamianie kodu
- [x] Auto-fix Code Runner - auto-install pakietów, retry loop
- [x] Self-healing code - automatyczne debugowanie przez LLM
- [x] Test-Driven Code Fixing - TDD z auto-generowaniem testów
- [x] Sandbox Environment - izolowane venv

### Conversation Engine
- [x] ConversationEngine - rozgałęzianie konwersacji
- [x] ThreadManager - równoległa obsługa wątków
- [x] ConversationBrancher - spawn sub-conversations
- [x] LLMAnalyzer - zastąpienie hardcoded patterns
- [x] Auto-Conversation API - `/api/code/auto-conversation`

### Autonomous Modules
- [x] Module Manager - tworzenie, budowanie, uruchamianie
- [x] LLM Module Generation - generowanie z opisu
- [x] Autonomous Agent - multi-step workflows
- [x] DSL Module Service - `module.create()`, `module.execute()`

### API & Backend
- [x] FastAPI server
- [x] WebSocket streaming - `/ws/chat`
- [x] API key authentication
- [x] Redis caching
- [x] Rate limiting (60 req/min default)
- [x] CORS support

### Frontend
- [x] Web Components - `<intent-form>`, `<intent-pay>`, etc.
- [x] JavaScript SDK v2.0
- [x] Zero-JS demo
- [x] Code block actions - Copy/Save/Run

### Infrastructure
- [x] Docker deployment
- [x] Docker Compose profiles (ollama, litellm)
- [x] Nginx reverse proxy
- [x] PostgreSQL + Redis

### CLI
- [x] CLI commands - dsl, dsl-call, services, repl
- [x] Config from .env
- [x] Default model selection

---

## 📊 Progress Tracking

| Kategoria | Done | In Progress | Planned | Total |
|-----------|------|-------------|---------|-------|
| Core | 7 | 0 | 0 | 7 |
| DSL | 6 | 0 | 0 | 6 |
| Code Exec | 5 | 0 | 0 | 5 |
| Conversation | 5 | 0 | 0 | 5 |
| Modules | 4 | 0 | 0 | 4 |
| API | 6 | 0 | 5 | 11 |
| Frontend | 4 | 0 | 8 | 12 |
| Infra | 4 | 0 | 5 | 9 |
| CLI | 3 | 0 | 5 | 8 |
| **Cleanup** | 0 | 0 | **6** | 6 |
| **Tests** | 0 | 0 | **7** | 7 |
| **CI/CD** | 0 | 0 | **3** | 3 |
| **Observability** | 0 | 0 | **5** | 5 |
| **Security** | 0 | 0 | **5** | 5 |
| **Total** | **44** | **0** | **49** | **93** |

---

## 🎯 Milestones

### v0.3.0 - Quality Release (Q1 2025)
- [ ] Wszystkie duplikaty usunięte
- [ ] 80%+ test coverage
- [ ] CI/CD pipeline działa
- [ ] Dokumentacja kompletna

### v0.4.0 - Production Ready (Q2 2025)
- [ ] Observability stack (Prometheus, Grafana)
- [ ] Security hardening
- [ ] Performance benchmarks
- [ ] GraphQL API

### v0.5.0 - Enterprise (Q3 2025)
- [ ] Multi-tenant
- [ ] RBAC
- [ ] SSO
- [ ] Audit trail

### v1.0.0 - GA (Q4 2025)
- [ ] Stable API
- [ ] Comprehensive docs
- [ ] SLA ready
- [ ] Commercial support

---

## 📝 Contributing

### Jak dodać nową funkcję
1. Utwórz issue na GitHub z opisem
2. Dodaj do odpowiedniej sekcji w tym pliku
3. Przypisz priorytet (🔴/🟡/🟢)
4. Zaimplementuj i przetestuj
5. Zaktualizuj dokumentację
6. Przenieś do sekcji "Ukończone"

### Priorytety
- 🔴 **Krytyczne** - Blocker, wymaga natychmiastowej uwagi
- 🟡 **Średnie** - Ważne ulepszenia UX/DX
- 🟢 **Niskie** - Nice-to-have, gdy będzie czas

### Code Style
```bash
# Lint
ruff check .
ruff format .

# Type check
mypy intentforge/

# Tests
pytest tests/ -v --cov=intentforge
```

### Pull Request Checklist
- [ ] Testy przechodzą
- [ ] Lint/type check OK
- [ ] Dokumentacja zaktualizowana
- [ ] TODO.md zaktualizowane
- [ ] CHANGELOG.md zaktualizowany

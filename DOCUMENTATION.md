# IntentForge - Dokumentacja

## 📚 Spis Treści

1. [Wprowadzenie](#wprowadzenie)
2. [Szybki Start](#szybki-start)
3. [Architektura](#architektura)
4. [SDK JavaScript](#sdk-javascript)
5. [Przykłady Użycia](#przykłady-użycia)
6. [Konfiguracja](#konfiguracja)
7. [API Reference](#api-reference)
8. [Struktura Plików](#struktura-plików)

---

## 🚀 Wprowadzenie

**IntentForge** to framework do dynamicznego generowania backendu z naturalnego języka. Pozwala tworzyć funkcjonalne aplikacje webowe używając tylko statycznego HTML i JavaScript.

### Główne cechy:

- **Zero Backend Setup** - nie musisz pisać backendu, IntentForge generuje go dynamicznie
- **MQTT Communication** - uniwersalny protokół działający z dowolnego frontendu
- **Integracje** - PayPal, Stripe, SMTP, kamery RTSP, bazy danych
- **Cachowanie** - generowany kod jest cachowany, kolejne wywołania są natychmiastowe
- **Bezpieczeństwo** - 3-poziomowa walidacja kodu, parametryzowane zapytania SQL

---

## ⚡ Szybki Start

### 1. Docker (najszybszy)

```bash
# Klonuj repo
git clone https://github.com/softreck/intentforge
cd intentforge

# Konfiguracja
cp .env.complete.example .env
# Edytuj .env - ustaw co najmniej ANTHROPIC_API_KEY i SMTP_*

# Uruchom
docker-compose up -d

# Otwórz demo
open http://localhost/examples/usecases/01_contact_form.html
```

### 2. Z JavaScript SDK (najprostszy)

```html
<!-- Dodaj SDK do strony -->
<script src="https://cdn.intentforge.io/v1/intentforge.min.js"
        data-broker="ws://localhost:9001"
        data-auto-bind="true">
</script>

<!-- Formularz z automatyczną obsługą -->
<form data-intent="contact">
    <input name="email" type="email" required>
    <textarea name="message" required></textarea>
    <button type="submit">Wyślij</button>
</form>
```

### 3. Z Python (programistyczny)

```python
from intentforge import generate, crud, form

# Generuj API z opisu
code = generate("Create REST API for products with pagination")

# Lub kompletny CRUD
files = crud("users")

# Lub formularz z backendem
files = form("contact", [
    {"name": "email", "type": "email", "required": True},
    {"name": "message", "type": "textarea"}
])
```

---

## 🏗️ Architektura

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND (Static HTML/JS)                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  IntentForge.js SDK                                      │  │
│  │  • api.form('contact').submit(data)                      │  │
│  │  • api.payment.checkout({amount: 49.99})                 │  │
│  │  • api.camera('rtsp://...').onMotion(callback)          │  │
│  │  • api.image.refresh('#img', 5000)                       │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────────┘
                              │ MQTT / REST
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INTENTFORGE SERVER                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ MQTT Broker │  │ REST API    │  │ WebSocket   │             │
│  │ (Mosquitto) │  │ (FastAPI)   │  │ (Events)    │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         └────────────────┴────────────────┘                     │
│                          │                                       │
│  ┌───────────────────────┴───────────────────────┐              │
│  │              Service Router                    │              │
│  │  • FormService    • PaymentService            │              │
│  │  • EmailService   • CameraService             │              │
│  │  • DataService    • CodeGenerator             │              │
│  └───────────────────────┬───────────────────────┘              │
│                          │                                       │
│  ┌───────────────────────┴───────────────────────┐              │
│  │              External Services                 │              │
│  │  ┌─────┐  ┌──────┐  ┌────┐  ┌──────────┐    │              │
│  │  │SMTP │  │PayPal│  │ DB │  │ Claude   │    │              │
│  │  └─────┘  └──────┘  └────┘  │ API      │    │              │
│  │  ┌──────┐ ┌──────┐ ┌─────┐ └──────────┘    │              │
│  │  │Stripe│ │ P24  │ │Redis│                  │              │
│  │  └──────┘ └──────┘ └─────┘                  │              │
│  └───────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 SDK JavaScript

### Instalacja

```html
<!-- CDN -->
<script src="https://cdn.intentforge.io/v1/intentforge.min.js"></script>

<!-- Lub lokalnie -->
<script src="/sdk/intentforge.js"></script>
```

### Inicjalizacja

```javascript
// Automatyczna (z data-attributes)
<script src="intentforge.js" 
        data-broker="ws://localhost:9001"
        data-auto-bind="true">
</script>

// Lub manualna
const api = await IntentForge.connect('ws://localhost:9001', {
    debug: true
});
```

### API

#### Formularze

```javascript
// Automatyczne bindowanie wszystkich formularzy z data-intent
api.autoBindForms();

// Manualna obsługa
api.form('contact')
   .onSuccess(data => console.log('Sent!', data))
   .onError(err => console.error(err))
   .submit({name: 'Jan', email: 'jan@example.com'});
```

#### Płatności

```javascript
// PayPal checkout
const result = await api.payment.checkout({
    amount: 49.99,
    currency: 'PLN',
    product: 'ebook-python',
    email: 'customer@example.com',
    provider: 'paypal'  // lub 'stripe', 'przelewy24'
});

// Przekierowanie do płatności
window.location.href = result.redirect_url;

// Weryfikacja
const status = await api.payment.verify(paymentId);
```

#### Email

```javascript
await api.email.send({
    to: 'recipient@example.com',
    template: 'welcome',
    data: {name: 'Jan'}
});
```

#### Kamera

```javascript
// Analiza obrazu
const analysis = await api.camera('rtsp://192.168.1.100/stream')
    .analyze({detect: ['motion', 'person']});

// Subskrypcja zdarzeń
api.camera('front-door')
   .onMotion(data => alert('Motion detected!'))
   .onObject('person', data => sendNotification(data));

// Stream do elementu img
api.camera('rtsp://...').startStream('camera-feed', {interval: 1000});
```

#### Auto-odświeżanie obrazów

```javascript
// Odświeżaj obraz co 5 sekund
api.image.refresh('#stock-chart', 5000, '/api/charts/stock');

// Zatrzymaj
api.image.stopRefresh('#stock-chart');
```

#### Dane (CRUD)

```javascript
// Lista
const products = await api.data('products').list({limit: 10});

// Jeden rekord
const product = await api.data('products').get(123);

// Utwórz
const newProduct = await api.data('products').create({
    name: 'Widget',
    price: 29.99
});

// Aktualizuj
await api.data('products').update(123, {price: 24.99});

// Usuń
await api.data('products').delete(123);
```

#### Real-time Events

```javascript
// Subskrypcja zdarzeń
api.on('order:new', (order) => {
    updateDashboard(order);
});

api.on('camera:motion', (data) => {
    showAlert(data);
});
```

---

## 📝 Przykłady Użycia

### 1. Formularz kontaktowy z emailem

**Plik:** `examples/usecases/01_contact_form.html`

```html
<form data-intent="contact">
    <input name="name" required>
    <input name="email" type="email" required>
    <textarea name="message" required></textarea>
    <button type="submit">Wyślij</button>
</form>
```

**Co się dzieje:**
1. Użytkownik wypełnia formularz
2. SDK wysyła dane przez MQTT
3. Backend zapisuje do bazy danych
4. Backend wysyła email do admina (SMTP z .env)
5. Backend wysyła potwierdzenie do użytkownika

### 2. Sprzedaż e-booka z PayPal

**Plik:** `examples/usecases/02_ebook_payment.html`

```javascript
// Checkout
const result = await api.payment.checkout({
    amount: 49.99,
    currency: 'PLN',
    product: 'ebook-python',
    email: 'customer@example.com',
    metadata: {
        download_url: '/downloads/ebook.pdf',
        send_email: true
    }
});

// Redirect to PayPal
window.location.href = result.redirect_url;
```

**Co się dzieje:**
1. Użytkownik klika "Kup"
2. SDK tworzy zamówienie przez PayPal API
3. Użytkownik jest przekierowany do PayPal
4. Po płatności wraca na stronę
5. Backend weryfikuje płatność
6. Backend wysyła email z linkiem do pobrania

### 3. Monitoring kamer z AI

**Plik:** `examples/usecases/03_camera_monitoring.html`

```javascript
// Auto-refresh kamery
api.image.refresh('#camera-main', 1000, '/api/camera/snapshot?source=front-door');

// Analiza co 5 sekund
setInterval(async () => {
    const result = await api.camera('front-door').analyze({
        detect: ['motion', 'person', 'vehicle']
    });
    
    if (result.detections.length > 0) {
        handleDetections(result.detections);
    }
}, 5000);

// Alerty email
api.on('camera:person', async (data) => {
    await api.email.send({
        to: 'security@company.com',
        template: 'camera_alert',
        data: data
    });
});
```

**Co się dzieje:**
1. Obraz z kamery RTSP odświeża się co sekundę
2. Co 5 sekund analiza AI (OpenCV DNN)
3. Wykrycie osoby → email z załącznikiem
4. Wszystko działa ze statycznej strony HTML

### 4. Dashboard z auto-odświeżaniem

**Plik:** `examples/usecases/04_realtime_dashboard.html`

```javascript
// Metryki co 5 sekund
setInterval(async () => {
    const metrics = await api._request('data', {
        action: 'query',
        table: 'metrics',
        type: 'realtime'
    });
    updateMetricsDisplay(metrics);
}, 5000);

// Obrazy z różnymi interwałami
api.image.refresh('#stock-chart', 5000);
api.image.refresh('#heatmap', 10000);
api.image.refresh('#weather-radar', 30000);

// Real-time events
api.on('order:new', (order) => {
    addOrderToTable(order);
});
```

---

## ⚙️ Konfiguracja

### Plik `.env`

Wszystkie usługi są konfigurowane przez zmienne środowiskowe:

```env
# Email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# Płatności
PAYPAL_CLIENT_ID=xxx
PAYPAL_SECRET=xxx
STRIPE_SECRET_KEY=sk_xxx

# Baza danych
DATABASE_URL=postgresql://user:pass@localhost/db

# Kamery
CAMERA_FRONT_DOOR=rtsp://192.168.1.100:554/stream1
```

Pełna lista w `.env.complete.example`.

---

## 📁 Struktura Plików

```
intentforge/
├── sdk/
│   └── intentforge.js          # JavaScript SDK dla frontendu
├── intentforge/
│   ├── __init__.py             # Eksporty
│   ├── simple.py               # One-liner API
│   ├── core.py                 # Intent, IntentForge
│   ├── generator.py            # Generowanie kodu (SQL, DOM, API)
│   ├── validator.py            # 3-poziomowa walidacja
│   ├── services.py             # Handlery usług (email, payment, camera)
│   ├── schema_registry.py      # Walidacja schematów JSON
│   ├── env_handler.py          # Obsługa .env
│   ├── patterns.py             # Fullstack patterns
│   ├── broker.py               # MQTT broker
│   ├── cache.py                # Cache (Redis/SQLite)
│   └── config.py               # Konfiguracja
├── examples/
│   ├── usecases/
│   │   ├── 01_contact_form.html        # Formularz kontaktowy
│   │   ├── 02_ebook_payment.html       # Sprzedaż z PayPal
│   │   ├── 03_camera_monitoring.html   # Monitoring kamer AI
│   │   └── 04_realtime_dashboard.html  # Dashboard real-time
│   ├── example1_oneliner.py    # Przykłady Python one-liner
│   ├── example2_static_html.html # Demo statyczne
│   └── example3_docker_workflow.py # Pełny workflow
├── config/
│   ├── mosquitto.conf          # MQTT broker config
│   └── nginx.conf              # Web server config
├── static/
│   └── index.html              # Landing page
├── docker-compose.yml          # Docker services
├── Dockerfile                  # Container image
├── Makefile                    # Build commands
├── .env.complete.example       # Kompletna konfiguracja
└── DOCUMENTATION.md            # Ten plik
```

---

## 🔧 Makefile Commands

```bash
make quickstart       # Pełna instalacja + konfiguracja
make demo             # Uruchom demo
make run-all          # Uruchom wszystkie serwisy
make generate-crud TABLE=users  # Generuj CRUD
make env-init         # Wygeneruj .env.example
make docker-compose-up  # Docker start
make test             # Testy
make help             # Lista komend
```

---

## 🔐 Bezpieczeństwo

1. **Walidacja 3-poziomowa:**
   - Syntax (AST parsing)
   - Security (dangerous patterns)
   - Semantic (logic verification)

2. **Parametryzowane SQL:**
   - Nigdy bezpośrednie wartości w SQL
   - Zawsze `%(param)s` placeholders

3. **Sandbox:**
   - Generowany kod uruchamiany w sandbox
   - Ograniczony dostęp do systemu

4. **Secrets:**
   - Wszystkie sekrety w `.env`
   - Nigdy w kodzie lub logach

---

## 📞 Wsparcie

- **GitHub Issues:** https://github.com/softreck/intentforge/issues
- **Dokumentacja:** https://intentforge.readthedocs.io
- **Email:** support@softreck.dev

---

## 📜 Licencja

MIT License - możesz używać komercyjnie.

---

**IntentForge v0.1.0** | Made with ❤️ by Softreck

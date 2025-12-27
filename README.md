# IntentForge 🚀

**Intent-Driven Development Framework**

IntentForge allows you to build software by describing **what** you want, not **how** to implement it. It turns natural language descriptions into executable code dynamically, safely, and efficiently using LLMs.

> **[📚 Read the Full Documentation](docs/INDEX.md)**

## 🌟 How It Works

IntentForge acts as a bridge between your intent (Natural Language) and the system's execution.

1.  **Capture Intent**: You describe a task (e.g., "Sort this list of users by date").
2.  **Generate Code**: The **[Generator](intentforge/generator.py)** uses an LLM to write the precise Python/JS code for that task.
3.  **Validate & Fix**: The system attempts to compile/run the code. If it fails, it auto-corrects itself.
4.  **Execute**: The code runs in a **[Secure Sandbox](intentforge/executor.py)**, and the result is returned.

## 🚀 Frontend Auto-Handler

Turn any HTML element into a smart component without writing JavaScript logic.

### 1. Smart Forms ([View Source](examples/static/form.html))

Automatically handle form submissions. The `intent.js` library captures the data and sends it to the backend.

```html
<!-- No JavaScript needed! Just describe what to do. -->
<form data-intent="Validate email, log the message, and return a success confirmation.">
    <input type="email" name="user_email">
    <button type="submit">Contact Support</button>
</form>
```

### 2. Dynamic Tables ([View Source](examples/static/table.html))

Populate tables automatically on page load.

```html
<!-- The table fills itself based on the intent -->
<table intent="Fetch 5 latest active users with their role and last login date.">
    <thead>
        <tr><th>Name</th><th>Role</th><th>Date</th></tr>
    </thead>
    <tbody></tbody>
</table>
```

## 🛠️ Key Components

- **[Server](examples/server.py)**: The FastAPI backend that processes intents.
- **[Intent Library](examples/static/intent.js)**: The lightweight JS client that brings HTML to life.
- **[Core Engine](intentforge/core.py)**: The heart of the framework managing the generation loop.

## 📦 Quick Start

### Option A: Python (Local)

1.  **Install**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Server**:
    ```bash
    python3 examples/server.py
    ```
3.  **Visit**: `http://localhost:8085/form.html`

### Option B: Docker (Recommended)

1.  **Start Services**:
    ```bash
    make start
    ```
2.  **Visit**: `http://localhost/examples/static/form.html`

## 📚 Documentation Index

- **[🏠 Home](docs/INDEX.md)**: Main entry point.
- **[⚙️ Setup](docs/setup.md)**: Installation and configuration details.
- **[🛠️ Usage](docs/usage.md)**: Deep dive into the Frontend Auto-Handler.
- **[🏗️ Architecture](docs/architecture.md)**: How the pieces fit together.
- **[📚 API](docs/api.md)**: Internal Python API reference.

## License

MIT

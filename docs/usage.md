# Usage Guide

IntentForge supports both traditional backend usage and a modern **Frontend Auto-Handler**.

## Frontend Auto-Handler

This feature allows you to make HTML elements "smart" by simply adding an `intent` attribute.

### Components

- **`intent.js`**: Client-side library ([Source](../examples/static/intent.js))
- **Backend API**: `POST /api/intent` ([Source](../examples/server.py))

### Examples

#### 1. Contact Form

Automatically handle form submissions without writing custom submit handlers.

```html
<form id="contact-form"
      data-intent="Handle contact form submission: Validate email and log message."
      data-intent-reset="true">

    <input type="email" name="email" required>
    <textarea name="message" required></textarea>

    <button type="submit">Send</button>
</form>
```

#### 2. Smart Table

Automatically populate a table with data fetched or generated via intent.

```html
<table class="table"
       intent="Return a list of 5 active mock users. Each user should have an id, name, and role.">
    <thead>
        <tr><th>ID</th><th>Name</th><th>Role</th></tr>
    </thead>
    <tbody>
        <!-- Populated automatically -->
    </tbody>
</table>
```

### Running the Example

### Running the Example

**Option A: Python (Local)**
1. Start the server:
   ```bash
   python3 examples/server.py
   ```
2. Navigate to `http://localhost:8085/form.html` or `http://localhost:8085/table.html`.

**Option B: Docker**
1. Start services:
   ```bash
   make start
   ```
2. Navigate to:
   - Form: `http://localhost/examples/static/form.html`
   - Table: `http://localhost/examples/static/table.html`

---

## Developer API

You can also use IntentForge directly in your Python code.

```python
import asyncio
from intentforge import IntentForge, Intent, IntentType

async def main():
    # Initialize
    forge = IntentForge(provider="ollama", model="llama3.1:8b")

    # Define Intent
    intent = Intent(
        description="Calculate the 10th Fibonacci number",
        intent_type=IntentType.WORKFLOW
    )

    # Process
    result = await forge.process_intent(intent)

    print(f"Result: {result.execution_result}")

if __name__ == "__main__":
    asyncio.run(main())
```

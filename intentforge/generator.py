"""
Code Generator - LLM-powered code generation with DSL support
Generates SQL, DOM manipulation, API endpoints, and firmware code
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .config import LLMSettings, get_settings
from .core import Intent, IntentType, TargetPlatform

logger = logging.getLogger(__name__)


class DSLType(Enum):
    """Supported DSL types for code generation"""

    SQL_QUERY = "sql_query"
    SQL_SCHEMA = "sql_schema"
    DOM_QUERY = "dom_query"
    DOM_MANIPULATION = "dom_manipulation"
    API_ENDPOINT = "api_endpoint"
    EVENT_HANDLER = "event_handler"
    FORM_HANDLER = "form_handler"
    DATABASE_CRUD = "database_crud"


@dataclass
class GenerationContext:
    """Context for code generation"""

    project_name: str = "app"
    database_type: str = "postgresql"
    framework: str = "fastapi"
    use_orm: bool = True
    orm_type: str = "sqlalchemy"
    auth_type: str | None = None
    env_prefix: str = ""

    # Schema information
    tables: dict[str, dict[str, str]] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)

    def to_prompt_context(self) -> str:
        """Convert to string for LLM prompt"""
        lines = [
            f"Project: {self.project_name}",
            f"Database: {self.database_type}",
            f"Framework: {self.framework}",
            f"ORM: {self.orm_type if self.use_orm else 'raw SQL'}",
        ]

        if self.tables:
            lines.append("\nAvailable tables:")
            for table, columns in self.tables.items():
                cols = ", ".join(f"{k}: {v}" for k, v in columns.items())
                lines.append(f"  - {table}({cols})")

        return "\n".join(lines)


class DSLGenerator(ABC):
    """Base class for DSL-specific generators"""

    @abstractmethod
    def generate(self, intent: Intent, context: GenerationContext) -> str:
        pass

    @abstractmethod
    def validate_output(self, code: str) -> tuple[bool, list[str]]:
        pass


class SQLGenerator(DSLGenerator):
    """Generate SQL queries and schemas"""

    # SQL templates for common operations
    TEMPLATES = {
        "select": """SELECT {columns}
FROM {table}
{joins}
WHERE {conditions}
{group_by}
{order_by}
{limit}""",
        "insert": """INSERT INTO {table} ({columns})
VALUES ({values})
RETURNING *""",
        "update": """UPDATE {table}
SET {assignments}
WHERE {conditions}
RETURNING *""",
        "delete": """DELETE FROM {table}
WHERE {conditions}
RETURNING *""",
        "create_table": """CREATE TABLE IF NOT EXISTS {table} (
    id SERIAL PRIMARY KEY,
    {columns},
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)""",
    }

    def generate(self, intent: Intent, context: GenerationContext) -> str:
        """Generate SQL from intent"""
        description = intent.description.lower()

        # Detect operation type
        if any(
            word in description for word in ["select", "get", "fetch", "find", "list", "pobierz"]
        ):
            return self._generate_select(intent, context)
        elif any(word in description for word in ["insert", "add", "create", "dodaj", "utwórz"]):
            return self._generate_insert(intent, context)
        elif any(
            word in description for word in ["update", "modify", "change", "zmień", "aktualizuj"]
        ):
            return self._generate_update(intent, context)
        elif any(word in description for word in ["delete", "remove", "usuń", "skasuj"]):
            return self._generate_delete(intent, context)
        elif any(word in description for word in ["schema", "table", "create table", "tabela"]):
            return self._generate_schema(intent, context)

        return self._generate_generic(intent, context)

    def _generate_select(self, intent: Intent, context: GenerationContext) -> str:
        """Generate SELECT query"""
        table = intent.context.get("table", "items")
        columns = intent.context.get("columns", "*")

        # Parse conditions from description
        conditions = self._extract_conditions(intent.description)

        # Build parameterized query
        query = f"""-- Auto-generated SELECT query
-- Intent: {intent.description}

SELECT {columns}
FROM {table}
WHERE {conditions}
ORDER BY created_at DESC
LIMIT %(limit)s OFFSET %(offset)s;

-- Parameters: {{limit: 50, offset: 0}}
"""
        return query

    def _generate_insert(self, intent: Intent, context: GenerationContext) -> str:
        """Generate INSERT query with parameters"""
        table = intent.context.get("table", "items")
        fields = intent.context.get("fields", ["name", "value"])

        columns = ", ".join(fields)
        placeholders = ", ".join(f"%({f})s" for f in fields)

        return f"""-- Auto-generated INSERT query
-- Intent: {intent.description}

INSERT INTO {table} ({columns})
VALUES ({placeholders})
RETURNING *;

-- Example parameters: {{{", ".join(f'{f}: "<value>"' for f in fields)}}}
"""

    def _generate_update(self, intent: Intent, context: GenerationContext) -> str:
        """Generate UPDATE query with safeguards"""
        table = intent.context.get("table", "items")
        fields = intent.context.get("fields", ["name"])

        assignments = ", ".join(f"{f} = %({f})s" for f in fields)

        return f"""-- Auto-generated UPDATE query
-- Intent: {intent.description}
-- WARNING: Always include WHERE clause to avoid updating all rows

UPDATE {table}
SET {assignments},
    updated_at = CURRENT_TIMESTAMP
WHERE id = %(id)s
RETURNING *;

-- Required parameter: id
-- Example: {{id: 1, {", ".join(f'{f}: "<value>"' for f in fields)}}}
"""

    def _generate_delete(self, intent: Intent, context: GenerationContext) -> str:
        """Generate DELETE query with safeguards"""
        table = intent.context.get("table", "items")

        return f"""-- Auto-generated DELETE query
-- Intent: {intent.description}
-- WARNING: Always verify WHERE clause before execution

DELETE FROM {table}
WHERE id = %(id)s
RETURNING *;

-- Required parameter: id
-- Soft delete alternative:
-- UPDATE {table} SET deleted_at = CURRENT_TIMESTAMP WHERE id = %(id)s;
"""

    def _generate_schema(self, intent: Intent, context: GenerationContext) -> str:
        """Generate CREATE TABLE schema"""
        table = intent.context.get("table", "items")
        fields = intent.context.get(
            "fields",
            {
                "name": "VARCHAR(255) NOT NULL",
                "description": "TEXT",
                "status": "VARCHAR(50) DEFAULT 'active'",
            },
        )

        columns = ",\n    ".join(f"{k} {v}" for k, v in fields.items())

        return f"""-- Auto-generated schema
-- Intent: {intent.description}

CREATE TABLE IF NOT EXISTS {table} (
    id SERIAL PRIMARY KEY,
    {columns},
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add indexes for common queries
CREATE INDEX IF NOT EXISTS idx_{table}_created_at ON {table}(created_at);

-- Add trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_{table}_updated_at
    BEFORE UPDATE ON {table}
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
"""

    def _generate_generic(self, intent: Intent, context: GenerationContext) -> str:
        """Generate generic SQL based on intent"""
        return f"""-- Intent: {intent.description}
-- Context: {json.dumps(intent.context, indent=2)}

-- TODO: Implement specific query based on intent
SELECT 1;
"""

    def _extract_conditions(self, description: str) -> str:
        """Extract WHERE conditions from natural language"""
        # Simple extraction - in production, use NLP
        conditions = []

        patterns = [
            (r'where\s+(\w+)\s*=\s*["\']?(\w+)["\']?', r"\1 = %(\1)s"),
            (r"with\s+(\w+)\s+(\w+)", r"\1 = %(\1)s"),
            (r"by\s+(\w+)", r"\1 = %(\1)s"),
        ]

        for pattern, replacement in patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                conditions.append(re.sub(pattern, replacement, match.group(0)))

        return " AND ".join(conditions) if conditions else "1=1"

    def validate_output(self, code: str) -> tuple[bool, list[str]]:
        """Validate SQL output"""
        errors = []

        # Check for dangerous patterns
        dangerous = [
            (r"DROP\s+TABLE", "DROP TABLE detected"),
            (r"TRUNCATE", "TRUNCATE detected"),
            (r"DELETE\s+FROM\s+\w+\s*;", "DELETE without WHERE"),
            (r"UPDATE\s+\w+\s+SET.*;\s*$", "UPDATE without WHERE"),
        ]

        for pattern, message in dangerous:
            if re.search(pattern, code, re.IGNORECASE):
                errors.append(message)

        # Check for parameterized queries
        if re.search(r"['\"].*\+.*['\"]", code):
            errors.append("String concatenation in query - use parameterized queries")

        return len(errors) == 0, errors


class DOMGenerator(DSLGenerator):
    """Generate DOM manipulation code"""

    def generate(self, intent: Intent, context: GenerationContext) -> str:
        """Generate DOM manipulation code"""
        description = intent.description.lower()

        if "form" in description:
            return self._generate_form_handler(intent, context)
        elif "fetch" in description or "api" in description:
            return self._generate_fetch(intent, context)
        elif "event" in description or "click" in description:
            return self._generate_event_handler(intent, context)

        return self._generate_generic_dom(intent, context)

    def _generate_form_handler(self, intent: Intent, context: GenerationContext) -> str:
        """Generate form handling code"""
        form_id = intent.context.get("form_id", "myForm")
        endpoint = intent.context.get("endpoint", "/api/submit")
        fields = intent.context.get("fields", ["name", "email"])

        field_extraction = "\n        ".join(
            f"const {f} = form.querySelector('[name=\"{f}\"]')?.value?.trim();" for f in fields
        )

        validation = "\n        ".join(
            f'if (!{f}) errors.push("{f.capitalize()} is required");' for f in fields
        )

        return f"""/**
 * Auto-generated form handler
 * Intent: {intent.description}
 * Generated for form: #{form_id}
 */

class FormHandler {{
    constructor(formSelector, endpoint) {{
        this.form = document.querySelector(formSelector);
        this.endpoint = endpoint;
        this.init();
    }}

    init() {{
        if (!this.form) {{
            console.error('Form not found');
            return;
        }}

        this.form.addEventListener('submit', (e) => this.handleSubmit(e));
    }}

    async handleSubmit(event) {{
        event.preventDefault();

        const form = this.form;
        const errors = [];

        // Extract form data
        {field_extraction}

        // Validation
        {validation}

        if (errors.length > 0) {{
            this.showErrors(errors);
            return;
        }}

        // Prepare data
        const data = {{
            {", ".join(f"{f}" for f in fields)}
        }};

        try {{
            const response = await fetch(this.endpoint, {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json',
                }},
                body: JSON.stringify(data)
            }});

            if (!response.ok) {{
                throw new Error(`HTTP ${{response.status}}`);
            }}

            const result = await response.json();
            this.onSuccess(result);

        }} catch (error) {{
            this.onError(error);
        }}
    }}

    showErrors(errors) {{
        const container = this.form.querySelector('.errors')
            || document.createElement('div');
        container.className = 'errors';
        container.innerHTML = errors.map(e => `<p class="error">${{e}}</p>`).join('');

        if (!this.form.contains(container)) {{
            this.form.prepend(container);
        }}
    }}

    onSuccess(result) {{
        console.log('Success:', result);
        this.form.reset();
        // Emit custom event
        this.form.dispatchEvent(new CustomEvent('formSuccess', {{ detail: result }}));
    }}

    onError(error) {{
        console.error('Error:', error);
        this.showErrors(['Submission failed. Please try again.']);
    }}
}}

// Initialize
document.addEventListener('DOMContentLoaded', () => {{
    new FormHandler('#{form_id}', '{endpoint}');
}});
"""

    def _generate_fetch(self, intent: Intent, context: GenerationContext) -> str:
        """Generate fetch/API call code"""
        endpoint = intent.context.get("endpoint", "/api/data")
        method = intent.context.get("method", "GET").upper()

        return f"""/**
 * Auto-generated API client
 * Intent: {intent.description}
 */

class APIClient {{
    constructor(baseUrl = '') {{
        this.baseUrl = baseUrl;
    }}

    async request(endpoint, options = {{}}) {{
        const url = `${{this.baseUrl}}${{endpoint}}`;

        const defaultOptions = {{
            headers: {{
                'Content-Type': 'application/json',
            }},
        }};

        // Merge options
        const config = {{
            ...defaultOptions,
            ...options,
            headers: {{
                ...defaultOptions.headers,
                ...options.headers,
            }},
        }};

        try {{
            const response = await fetch(url, config);

            if (!response.ok) {{
                const error = await response.json().catch(() => ({{}}));
                throw new APIError(response.status, error.message || 'Request failed');
            }}

            return await response.json();

        }} catch (error) {{
            if (error instanceof APIError) throw error;
            throw new APIError(0, error.message);
        }}
    }}

    // Convenience methods
    get(endpoint, params = {{}}) {{
        const query = new URLSearchParams(params).toString();
        const url = query ? `${{endpoint}}?${{query}}` : endpoint;
        return this.request(url, {{ method: 'GET' }});
    }}

    post(endpoint, data) {{
        return this.request(endpoint, {{
            method: 'POST',
            body: JSON.stringify(data),
        }});
    }}

    put(endpoint, data) {{
        return this.request(endpoint, {{
            method: 'PUT',
            body: JSON.stringify(data),
        }});
    }}

    delete(endpoint) {{
        return this.request(endpoint, {{ method: 'DELETE' }});
    }}
}}

class APIError extends Error {{
    constructor(status, message) {{
        super(message);
        this.status = status;
        this.name = 'APIError';
    }}
}}

// Usage example
const api = new APIClient('');

// {method} {endpoint}
api.{method.lower()}('{endpoint}')
    .then(data => console.log(data))
    .catch(err => console.error(err));
"""

    def _generate_event_handler(self, intent: Intent, context: GenerationContext) -> str:
        """Generate event handler code"""
        selector = intent.context.get("selector", ".button")
        event = intent.context.get("event", "click")

        return f"""/**
 * Auto-generated event handler
 * Intent: {intent.description}
 */

class EventManager {{
    constructor() {{
        this.handlers = new Map();
    }}

    on(selector, event, handler, options = {{}}) {{
        const elements = document.querySelectorAll(selector);

        elements.forEach(el => {{
            const wrappedHandler = (e) => {{
                try {{
                    handler.call(el, e, el);
                }} catch (error) {{
                    console.error(`Event handler error for ${{event}} on ${{selector}}:`, error);
                }}
            }};

            el.addEventListener(event, wrappedHandler, options);

            // Store for cleanup
            const key = `${{selector}}:${{event}}`;
            if (!this.handlers.has(key)) {{
                this.handlers.set(key, []);
            }}
            this.handlers.get(key).push({{ el, handler: wrappedHandler }});
        }});

        return this;
    }}

    off(selector, event) {{
        const key = `${{selector}}:${{event}}`;
        const handlers = this.handlers.get(key) || [];

        handlers.forEach(({{ el, handler }}) => {{
            el.removeEventListener(event, handler);
        }});

        this.handlers.delete(key);
        return this;
    }}

    // Delegate events (for dynamic elements)
    delegate(containerSelector, targetSelector, event, handler) {{
        const container = document.querySelector(containerSelector);

        if (!container) return this;

        container.addEventListener(event, (e) => {{
            const target = e.target.closest(targetSelector);
            if (target && container.contains(target)) {{
                handler.call(target, e, target);
            }}
        }});

        return this;
    }}
}}

// Initialize
const events = new EventManager();

// Example: Handle {event} on {selector}
events.on('{selector}', '{event}', function(e, element) {{
    console.log('{event} event on:', element);

    // Your handler logic here
    // Intent: {intent.description}
}});
"""

    def _generate_generic_dom(self, intent: Intent, context: GenerationContext) -> str:
        return f"""/**
 * DOM manipulation helper
 * Intent: {intent.description}
 */

const DOM = {{
    select: (selector) => document.querySelector(selector),
    selectAll: (selector) => [...document.querySelectorAll(selector)],

    create: (tag, attrs = {{}}, children = []) => {{
        const el = document.createElement(tag);
        Object.entries(attrs).forEach(([k, v]) => {{
            if (k === 'class') el.className = v;
            else if (k === 'style' && typeof v === 'object') {{
                Object.assign(el.style, v);
            }} else {{
                el.setAttribute(k, v);
            }}
        }});
        children.forEach(child => {{
            if (typeof child === 'string') {{
                el.appendChild(document.createTextNode(child));
            }} else {{
                el.appendChild(child);
            }}
        }});
        return el;
    }},

    append: (parent, ...children) => {{
        const p = typeof parent === 'string' ? DOM.select(parent) : parent;
        children.forEach(c => p?.appendChild(c));
        return p;
    }},

    remove: (selector) => {{
        const el = DOM.select(selector);
        el?.parentNode?.removeChild(el);
    }},

    html: (selector, content) => {{
        const el = DOM.select(selector);
        if (el) el.innerHTML = content;
        return el;
    }},

    text: (selector, content) => {{
        const el = DOM.select(selector);
        if (el) el.textContent = content;
        return el;
    }}
}};
"""

    def validate_output(self, code: str) -> tuple[bool, list[str]]:
        """Validate DOM code"""
        errors = []

        # Check for dangerous patterns
        if "eval(" in code:
            errors.append("eval() detected - security risk")
        if "innerHTML" in code and "user" in code.lower():
            errors.append("innerHTML with user input - XSS risk")
        if "document.write" in code:
            errors.append("document.write() is deprecated")

        return len(errors) == 0, errors


class APIEndpointGenerator(DSLGenerator):
    """Generate API endpoint code"""

    def generate(self, intent: Intent, context: GenerationContext) -> str:
        """Generate API endpoint based on intent"""
        framework = context.framework.lower()

        if framework == "fastapi":
            return self._generate_fastapi(intent, context)
        elif framework == "flask":
            return self._generate_flask(intent, context)
        elif framework in ("express", "nodejs"):
            return self._generate_express(intent, context)

        return self._generate_fastapi(intent, context)  # Default

    def _generate_fastapi(self, intent: Intent, context: GenerationContext) -> str:
        """Generate FastAPI endpoint"""
        endpoint = intent.context.get("endpoint", "/api/items")
        method = intent.context.get("method", "GET").upper()
        model_name = intent.context.get("model", "Item")

        methods_code = {
            "GET": self._fastapi_get(endpoint, model_name, intent),
            "POST": self._fastapi_post(endpoint, model_name, intent),
            "PUT": self._fastapi_put(endpoint, model_name, intent),
            "DELETE": self._fastapi_delete(endpoint, model_name, intent),
        }

        return f'''"""
Auto-generated FastAPI endpoint
Intent: {intent.description}
"""

import os
from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from .env
DATABASE_URL = os.getenv("DB_HOST", "localhost")
DATABASE_NAME = os.getenv("DB_NAME", "app")
DATABASE_USER = os.getenv("DB_USER", "postgres")
DATABASE_PASSWORD = os.getenv("DB_PASSWORD", "")

router = APIRouter(prefix="/api", tags=["{model_name.lower()}s"])


# Pydantic models
class {model_name}Base(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None

    class Config:
        from_attributes = True


class {model_name}Create({model_name}Base):
    pass


class {model_name}Update(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None


class {model_name}Response({model_name}Base):
    id: int
    created_at: datetime
    updated_at: datetime


class {model_name}ListResponse(BaseModel):
    items: List[{model_name}Response]
    total: int
    page: int
    page_size: int


# Database dependency
def get_db():
    """Database session dependency"""
    from .database import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


{methods_code.get(method, methods_code["GET"])}


# Additional utility endpoints
@router.get("{endpoint}/health")
async def health_check():
    """Health check endpoint"""
    return {{"status": "healthy", "timestamp": datetime.utcnow()}}
'''

    def _fastapi_get(self, endpoint: str, model: str, intent: Intent) -> str:
        return f'''
@router.get("{endpoint}", response_model={model}ListResponse)
async def list_{model.lower()}s(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    List {model.lower()}s with pagination
    Intent: {intent.description}
    """
    from .models import {model}Model

    query = db.query({model}Model)

    if search:
        query = query.filter({model}Model.name.ilike(f"%{{search}}%"))

    total = query.count()
    items = query.offset((page - 1) * page_size).limit(page_size).all()

    return {{
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size
    }}


@router.get("{endpoint}/{{item_id}}", response_model={model}Response)
async def get_{model.lower()}(
    item_id: int,
    db: Session = Depends(get_db)
):
    """Get single {model.lower()} by ID"""
    from .models import {model}Model

    item = db.query({model}Model).filter({model}Model.id == item_id).first()

    if not item:
        raise HTTPException(status_code=404, detail="{model} not found")

    return item
'''

    def _fastapi_post(self, endpoint: str, model: str, intent: Intent) -> str:
        return f'''
@router.post("{endpoint}", response_model={model}Response, status_code=201)
async def create_{model.lower()}(
    data: {model}Create,
    db: Session = Depends(get_db)
):
    """
    Create new {model.lower()}
    Intent: {intent.description}
    """
    from .models import {model}Model

    item = {model}Model(**data.model_dump())
    db.add(item)
    db.commit()
    db.refresh(item)

    return item
'''

    def _fastapi_put(self, endpoint: str, model: str, intent: Intent) -> str:
        return f'''
@router.put("{endpoint}/{{item_id}}", response_model={model}Response)
async def update_{model.lower()}(
    item_id: int,
    data: {model}Update,
    db: Session = Depends(get_db)
):
    """
    Update {model.lower()}
    Intent: {intent.description}
    """
    from .models import {model}Model

    item = db.query({model}Model).filter({model}Model.id == item_id).first()

    if not item:
        raise HTTPException(status_code=404, detail="{model} not found")

    update_data = data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(item, key, value)

    item.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(item)

    return item
'''

    def _fastapi_delete(self, endpoint: str, model: str, intent: Intent) -> str:
        return f'''
@router.delete("{endpoint}/{{item_id}}", status_code=204)
async def delete_{model.lower()}(
    item_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete {model.lower()}
    Intent: {intent.description}
    """
    from .models import {model}Model

    item = db.query({model}Model).filter({model}Model.id == item_id).first()

    if not item:
        raise HTTPException(status_code=404, detail="{model} not found")

    db.delete(item)
    db.commit()

    return None
'''

    def _generate_flask(self, intent: Intent, context: GenerationContext) -> str:
        """Generate Flask endpoint"""
        endpoint = intent.context.get("endpoint", "/api/items")
        method = intent.context.get("method", "GET")

        return f'''"""
Auto-generated Flask endpoint
Intent: {intent.description}
"""

import os
from flask import Blueprint, request, jsonify
from dotenv import load_dotenv

load_dotenv()

api = Blueprint('api', __name__, url_prefix='/api')

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///app.db")


@api.route('{endpoint}', methods=['{method}'])
def handle_{endpoint.replace("/", "_").strip("_")}():
    """
    {intent.description}
    """
    if request.method == 'GET':
        # Get list with pagination
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)

        # TODO: Implement database query
        items = []

        return jsonify({{
            'items': items,
            'page': page,
            'per_page': per_page
        }})

    elif request.method == 'POST':
        data = request.get_json()

        if not data:
            return jsonify({{'error': 'No data provided'}}), 400

        # TODO: Validate and save

        return jsonify(data), 201
'''

    def _generate_express(self, intent: Intent, context: GenerationContext) -> str:
        """Generate Express.js endpoint"""
        endpoint = intent.context.get("endpoint", "/api/items")

        return f"""/**
 * Auto-generated Express.js endpoint
 * Intent: {intent.description}
 */

require('dotenv').config();
const express = require('express');
const router = express.Router();

// Configuration from .env
const DB_HOST = process.env.DB_HOST || 'localhost';
const DB_NAME = process.env.DB_NAME || 'app';

router.get('{endpoint}', async (req, res) => {{
    try {{
        const {{ page = 1, limit = 50 }} = req.query;

        // TODO: Implement database query
        const items = [];

        res.json({{
            items,
            page: parseInt(page),
            limit: parseInt(limit)
        }});
    }} catch (error) {{
        res.status(500).json({{ error: error.message }});
    }}
}});

router.post('{endpoint}', async (req, res) => {{
    try {{
        const data = req.body;

        // TODO: Validate and save

        res.status(201).json(data);
    }} catch (error) {{
        res.status(500).json({{ error: error.message }});
    }}
}});

module.exports = router;
"""

    def validate_output(self, code: str) -> tuple[bool, list[str]]:
        errors = []

        # Check for hardcoded credentials
        if re.search(r'password\s*=\s*["\'][^"\']+["\']', code, re.IGNORECASE):
            if "os.getenv" not in code and "process.env" not in code:
                errors.append("Hardcoded password detected - use environment variables")

        return len(errors) == 0, errors


class CodeGenerator:
    """
    Main code generator - orchestrates DSL generators and LLM
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20250929",
        settings: LLMSettings | None = None,
    ):
        self.settings = settings or get_settings().llm
        self.api_key = api_key or self.settings.api_key.get_secret_value()
        self.model = model or self.settings.model

        # Initialize DSL generators
        self.generators: dict[DSLType, DSLGenerator] = {
            DSLType.SQL_QUERY: SQLGenerator(),
            DSLType.SQL_SCHEMA: SQLGenerator(),
            DSLType.DOM_QUERY: DOMGenerator(),
            DSLType.DOM_MANIPULATION: DOMGenerator(),
            DSLType.FORM_HANDLER: DOMGenerator(),
            DSLType.API_ENDPOINT: APIEndpointGenerator(),
            DSLType.DATABASE_CRUD: APIEndpointGenerator(),
        }

        self._client = None

    @property
    def client(self):
        """Lazy load Anthropic client"""
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    async def generate(
        self, intent: Intent, context: GenerationContext | None = None
    ) -> tuple[str, str]:
        """
        Generate code from intent
        Returns (code, language)
        """
        context = context or GenerationContext()

        # Determine DSL type from intent
        dsl_type = self._determine_dsl_type(intent)

        # Try local generator first
        generator = self.generators.get(dsl_type)
        if generator:
            try:
                code = generator.generate(intent, context)
                language = self._determine_language(intent, dsl_type)

                # Validate
                is_valid, errors = generator.validate_output(code)
                if not is_valid:
                    logger.warning(f"Generated code has issues: {errors}")

                return code, language
            except Exception as e:
                logger.warning(f"Local generator failed: {e}, falling back to LLM")

        # Fall back to LLM
        return await self._generate_with_llm(intent, context)

    def _determine_dsl_type(self, intent: Intent) -> DSLType:
        """Determine which DSL generator to use"""
        desc = intent.description.lower()
        intent_type = intent.intent_type

        if intent_type == IntentType.DATABASE_SCHEMA:
            return DSLType.SQL_SCHEMA

        if any(word in desc for word in ["sql", "query", "select", "insert", "database", "tabela"]):
            return DSLType.SQL_QUERY

        if any(word in desc for word in ["form", "formularz", "submit"]):
            return DSLType.FORM_HANDLER

        if any(word in desc for word in ["dom", "element", "html", "click", "event"]):
            return DSLType.DOM_MANIPULATION

        if any(word in desc for word in ["api", "endpoint", "rest", "route"]):
            return DSLType.API_ENDPOINT

        if any(word in desc for word in ["crud", "resource"]):
            return DSLType.DATABASE_CRUD

        return DSLType.API_ENDPOINT  # Default

    def _determine_language(self, intent: Intent, dsl_type: DSLType) -> str:
        """Determine output language"""
        platform = intent.target_platform

        if dsl_type in (DSLType.SQL_QUERY, DSLType.SQL_SCHEMA):
            return "sql"

        if dsl_type in (DSLType.DOM_MANIPULATION, DSLType.DOM_QUERY, DSLType.FORM_HANDLER):
            return "javascript"

        platform_languages = {
            TargetPlatform.PYTHON_FASTAPI: "python",
            TargetPlatform.PYTHON_FLASK: "python",
            TargetPlatform.NODEJS_EXPRESS: "javascript",
            TargetPlatform.ARDUINO_CPP: "cpp",
            TargetPlatform.ESP32_MICROPYTHON: "python",
            TargetPlatform.JETSON_PYTHON: "python",
            TargetPlatform.RASPBERRY_PI: "python",
        }

        return platform_languages.get(platform, "python")

    async def _generate_with_llm(
        self, intent: Intent, context: GenerationContext
    ) -> tuple[str, str]:
        """Generate code using LLM"""

        system_prompt = f"""You are a code generator. Generate production-ready code based on the user's intent.

Context:
{context.to_prompt_context()}

Rules:
1. Generate ONLY code, no explanations
2. Use environment variables for secrets (from .env file)
3. Include proper error handling
4. Follow security best practices
5. Use parameterized queries for SQL
6. Add appropriate comments

Target platform: {intent.target_platform.value}
"""

        user_prompt = f"""Generate code for the following intent:

Intent: {intent.description}
Type: {intent.intent_type.value}
Context: {json.dumps(intent.context)}
Constraints: {intent.constraints}

Return only the code."""

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            code = message.content[0].text

            # Extract code from markdown if present
            code = self._extract_code(code)
            language = self._determine_language(intent, self._determine_dsl_type(intent))

            return code, language

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    def _extract_code(self, text: str) -> str:
        """Extract code from markdown code blocks"""
        # Try to find code block
        pattern = r"```(?:\w+)?\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)

        if match:
            return match.group(1).strip()

        return text.strip()

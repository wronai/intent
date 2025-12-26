"""
Fullstack Patterns - Common solutions for popular development problems
Provides pre-built patterns for form handling, CRUD, auth, and more
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json


class PatternType(Enum):
    """Available fullstack patterns"""
    FORM_TO_DATABASE = "form_to_database"
    CRUD_API = "crud_api"
    AUTH_JWT = "auth_jwt"
    FILE_UPLOAD = "file_upload"
    REALTIME_MQTT = "realtime_mqtt"
    PAGINATION = "pagination"
    SEARCH_FILTER = "search_filter"
    WEBSOCKET_EVENTS = "websocket_events"
    CACHE_LAYER = "cache_layer"
    RATE_LIMITING = "rate_limiting"


@dataclass
class PatternConfig:
    """Configuration for a pattern"""
    pattern_type: PatternType
    target_table: str = "items"
    fields: List[Dict[str, Any]] = field(default_factory=list)
    auth_required: bool = False
    use_validation: bool = True
    framework: str = "fastapi"
    include_tests: bool = True


class FullstackPatterns:
    """
    Collection of production-ready fullstack patterns
    """
    
    @staticmethod
    def form_to_database(config: PatternConfig) -> Dict[str, str]:
        """
        Complete form-to-database pattern
        Generates: frontend form, API endpoint, database model, validation
        """
        table = config.target_table
        fields = config.fields or [
            {"name": "name", "type": "text", "required": True},
            {"name": "email", "type": "email", "required": True},
            {"name": "message", "type": "textarea", "required": False}
        ]
        
        return {
            "frontend_html": FullstackPatterns._generate_form_html(table, fields),
            "frontend_js": FullstackPatterns._generate_form_js(table, fields),
            "backend_model": FullstackPatterns._generate_model(table, fields),
            "backend_api": FullstackPatterns._generate_api_endpoint(table, fields, config),
            "backend_schema": FullstackPatterns._generate_pydantic_schema(table, fields),
            "database_migration": FullstackPatterns._generate_migration(table, fields),
            "tests": FullstackPatterns._generate_tests(table, fields) if config.include_tests else ""
        }
    
    @staticmethod
    def _generate_form_html(table: str, fields: List[Dict]) -> str:
        """Generate HTML form"""
        field_html = []
        for f in fields:
            input_type = f.get("type", "text")
            required = 'required' if f.get("required") else ''
            
            if input_type == "textarea":
                field_html.append(f'''
    <div class="form-group">
        <label for="{f['name']}">{f['name'].replace('_', ' ').title()}</label>
        <textarea 
            id="{f['name']}" 
            name="{f['name']}" 
            class="form-control"
            {required}
        ></textarea>
    </div>''')
            elif input_type == "select":
                options = f.get("options", [])
                opts_html = '\n'.join(f'<option value="{o}">{o}</option>' for o in options)
                field_html.append(f'''
    <div class="form-group">
        <label for="{f['name']}">{f['name'].replace('_', ' ').title()}</label>
        <select id="{f['name']}" name="{f['name']}" class="form-control" {required}>
            <option value="">Select...</option>
            {opts_html}
        </select>
    </div>''')
            else:
                field_html.append(f'''
    <div class="form-group">
        <label for="{f['name']}">{f['name'].replace('_', ' ').title()}</label>
        <input 
            type="{input_type}" 
            id="{f['name']}" 
            name="{f['name']}" 
            class="form-control"
            {required}
        />
    </div>''')
        
        return f'''<!-- Auto-generated form for {table} -->
<!-- Sends data via MQTT or REST API -->

<form id="{table}-form" class="needs-validation" novalidate>
    {''.join(field_html)}
    
    <div class="form-group mt-3">
        <button type="submit" class="btn btn-primary">Submit</button>
        <button type="reset" class="btn btn-secondary">Reset</button>
    </div>
    
    <div id="form-feedback" class="alert" style="display: none;"></div>
</form>

<!-- Include intentforge client -->
<script src="/static/js/intentforge-client.js"></script>
<script src="/static/js/{table}-form.js"></script>
'''
    
    @staticmethod
    def _generate_form_js(table: str, fields: List[Dict]) -> str:
        """Generate JavaScript form handler with MQTT support"""
        field_names = [f['name'] for f in fields]
        
        return f'''/**
 * Auto-generated form handler for {table}
 * Supports both REST API and MQTT submission
 */

class {table.title().replace('_', '')}Form {{
    constructor(options = {{}}) {{
        this.formId = '{table}-form';
        this.apiEndpoint = options.apiEndpoint || '/api/{table}';
        this.useMQTT = options.useMQTT || false;
        this.mqttTopic = options.mqttTopic || 'intentforge/forms/{table}';
        
        this.form = document.getElementById(this.formId);
        this.feedback = document.getElementById('form-feedback');
        
        // Initialize MQTT if enabled
        if (this.useMQTT && window.IntentForgeMQTT) {{
            this.mqtt = new IntentForgeMQTT({{
                broker: options.mqttBroker || 'ws://localhost:9001'
            }});
        }}
        
        this.init();
    }}
    
    init() {{
        if (!this.form) return;
        
        this.form.addEventListener('submit', (e) => this.handleSubmit(e));
        
        // Add real-time validation
        const inputs = this.form.querySelectorAll('input, textarea, select');
        inputs.forEach(input => {{
            input.addEventListener('blur', () => this.validateField(input));
            input.addEventListener('input', () => this.clearError(input));
        }});
    }}
    
    async handleSubmit(e) {{
        e.preventDefault();
        
        if (!this.validateForm()) {{
            return;
        }}
        
        const formData = this.collectFormData();
        
        try {{
            this.setLoading(true);
            
            let result;
            if (this.useMQTT && this.mqtt) {{
                result = await this.submitViaMQTT(formData);
            }} else {{
                result = await this.submitViaAPI(formData);
            }}
            
            this.showSuccess('Data saved successfully!');
            this.form.reset();
            
            // Dispatch custom event
            this.form.dispatchEvent(new CustomEvent('{table}:saved', {{
                detail: result
            }}));
            
        }} catch (error) {{
            this.showError(error.message || 'An error occurred');
        }} finally {{
            this.setLoading(false);
        }}
    }}
    
    collectFormData() {{
        const data = {{}};
        const fields = {json.dumps(field_names)};
        
        fields.forEach(field => {{
            const input = this.form.querySelector(`[name="${{field}}"]`);
            if (input) {{
                data[field] = input.value;
            }}
        }});
        
        return data;
    }}
    
    async submitViaAPI(data) {{
        const response = await fetch(this.apiEndpoint, {{
            method: 'POST',
            headers: {{
                'Content-Type': 'application/json',
            }},
            body: JSON.stringify(data)
        }});
        
        if (!response.ok) {{
            const error = await response.json();
            throw new Error(error.detail || 'Server error');
        }}
        
        return response.json();
    }}
    
    async submitViaMQTT(data) {{
        return new Promise((resolve, reject) => {{
            const requestId = crypto.randomUUID();
            
            // Subscribe to response
            this.mqtt.subscribe(`${{this.mqttTopic}}/response/${{requestId}}`, (msg) => {{
                if (msg.success) {{
                    resolve(msg.data);
                }} else {{
                    reject(new Error(msg.error));
                }}
            }});
            
            // Publish request
            this.mqtt.publish(this.mqttTopic, {{
                request_id: requestId,
                action: 'create',
                data: data
            }});
            
            // Timeout
            setTimeout(() => {{
                reject(new Error('Request timeout'));
            }}, 30000);
        }});
    }}
    
    validateForm() {{
        let isValid = true;
        const inputs = this.form.querySelectorAll('[required]');
        
        inputs.forEach(input => {{
            if (!this.validateField(input)) {{
                isValid = false;
            }}
        }});
        
        return isValid;
    }}
    
    validateField(input) {{
        const value = input.value.trim();
        let isValid = true;
        let message = '';
        
        // Required check
        if (input.hasAttribute('required') && !value) {{
            isValid = false;
            message = 'This field is required';
        }}
        
        // Type-specific validation
        if (value && input.type === 'email') {{
            const emailRegex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
            if (!emailRegex.test(value)) {{
                isValid = false;
                message = 'Please enter a valid email';
            }}
        }}
        
        if (!isValid) {{
            this.showFieldError(input, message);
        }} else {{
            this.clearError(input);
        }}
        
        return isValid;
    }}
    
    showFieldError(input, message) {{
        input.classList.add('is-invalid');
        
        let feedback = input.parentElement.querySelector('.invalid-feedback');
        if (!feedback) {{
            feedback = document.createElement('div');
            feedback.className = 'invalid-feedback';
            input.parentElement.appendChild(feedback);
        }}
        feedback.textContent = message;
    }}
    
    clearError(input) {{
        input.classList.remove('is-invalid');
    }}
    
    showSuccess(message) {{
        this.feedback.className = 'alert alert-success';
        this.feedback.textContent = message;
        this.feedback.style.display = 'block';
    }}
    
    showError(message) {{
        this.feedback.className = 'alert alert-danger';
        this.feedback.textContent = message;
        this.feedback.style.display = 'block';
    }}
    
    setLoading(loading) {{
        const btn = this.form.querySelector('button[type="submit"]');
        if (btn) {{
            btn.disabled = loading;
            btn.innerHTML = loading 
                ? '<span class="spinner-border spinner-border-sm"></span> Saving...'
                : 'Submit';
        }}
    }}
}}

// Auto-initialize
document.addEventListener('DOMContentLoaded', () => {{
    window.{table}Form = new {table.title().replace('_', '')}Form();
}});
'''
    
    @staticmethod
    def _generate_model(table: str, fields: List[Dict]) -> str:
        """Generate SQLAlchemy model"""
        type_mapping = {
            "text": "String(255)",
            "email": "String(255)",
            "password": "String(255)",
            "number": "Integer",
            "float": "Float",
            "date": "Date",
            "datetime": "DateTime",
            "textarea": "Text",
            "boolean": "Boolean",
            "select": "String(100)"
        }
        
        columns = []
        for f in fields:
            col_type = type_mapping.get(f.get("type", "text"), "String(255)")
            nullable = "False" if f.get("required") else "True"
            columns.append(f'    {f["name"]} = Column({col_type}, nullable={nullable})')
        
        return f'''"""
SQLAlchemy model for {table}
Auto-generated by IntentForge
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, Date
from sqlalchemy.sql import func
from .database import Base


class {table.title().replace('_', '')}(Base):
    __tablename__ = "{table}"
    
    id = Column(Integer, primary_key=True, index=True)
{chr(10).join(columns)}
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def to_dict(self):
        return {{
            "id": self.id,
{chr(10).join(f'            "{f["name"]}": self.{f["name"]},' for f in fields)}
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }}
'''
    
    @staticmethod
    def _generate_api_endpoint(table: str, fields: List[Dict], config: PatternConfig) -> str:
        """Generate FastAPI endpoint"""
        field_names = [f['name'] for f in fields]
        
        auth_decorator = ""
        auth_import = ""
        auth_param = ""
        
        if config.auth_required:
            auth_import = "from .auth import get_current_user"
            auth_param = ", current_user = Depends(get_current_user)"
        
        return f'''"""
API endpoints for {table}
Auto-generated by IntentForge
"""

import os
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from .database import get_session
from .models import {table.title().replace('_', '')}
from .schemas import {table.title().replace('_', '')}Create, {table.title().replace('_', '')}Response, {table.title().replace('_', '')}Update
{auth_import}

router = APIRouter(prefix="/api/{table}", tags=["{table}"])


@router.get("/", response_model=List[{table.title().replace('_', '')}Response])
async def list_{table}(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    search: Optional[str] = None,
    session: AsyncSession = Depends(get_session){auth_param}
):
    """List all {table} with pagination and search"""
    query = select({table.title().replace('_', '')})
    
    if search:
        # Add search filter
        query = query.where(
            {table.title().replace('_', '')}.name.ilike(f"%{{search}}%")
        )
    
    query = query.offset(skip).limit(limit)
    result = await session.execute(query)
    return result.scalars().all()


@router.get("/{{item_id}}", response_model={table.title().replace('_', '')}Response)
async def get_{table.rstrip('s')}(
    item_id: int,
    session: AsyncSession = Depends(get_session){auth_param}
):
    """Get single {table} by ID"""
    result = await session.execute(
        select({table.title().replace('_', '')}).where({table.title().replace('_', '')}.id == item_id)
    )
    item = result.scalar_one_or_none()
    
    if not item:
        raise HTTPException(status_code=404, detail="{table} not found")
    
    return item


@router.post("/", response_model={table.title().replace('_', '')}Response, status_code=201)
async def create_{table.rstrip('s')}(
    data: {table.title().replace('_', '')}Create,
    session: AsyncSession = Depends(get_session){auth_param}
):
    """Create new {table}"""
    item = {table.title().replace('_', '')}(**data.model_dump())
    session.add(item)
    await session.commit()
    await session.refresh(item)
    return item


@router.put("/{{item_id}}", response_model={table.title().replace('_', '')}Response)
async def update_{table.rstrip('s')}(
    item_id: int,
    data: {table.title().replace('_', '')}Update,
    session: AsyncSession = Depends(get_session){auth_param}
):
    """Update existing {table}"""
    result = await session.execute(
        select({table.title().replace('_', '')}).where({table.title().replace('_', '')}.id == item_id)
    )
    item = result.scalar_one_or_none()
    
    if not item:
        raise HTTPException(status_code=404, detail="{table} not found")
    
    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(item, key, value)
    
    await session.commit()
    await session.refresh(item)
    return item


@router.delete("/{{item_id}}", status_code=204)
async def delete_{table.rstrip('s')}(
    item_id: int,
    session: AsyncSession = Depends(get_session){auth_param}
):
    """Delete {table}"""
    result = await session.execute(
        select({table.title().replace('_', '')}).where({table.title().replace('_', '')}.id == item_id)
    )
    item = result.scalar_one_or_none()
    
    if not item:
        raise HTTPException(status_code=404, detail="{table} not found")
    
    await session.delete(item)
    await session.commit()


@router.get("/count")
async def count_{table}(
    session: AsyncSession = Depends(get_session){auth_param}
):
    """Get total count of {table}"""
    result = await session.execute(
        select(func.count({table.title().replace('_', '')}.id))
    )
    return {{"count": result.scalar()}}
'''
    
    @staticmethod
    def _generate_pydantic_schema(table: str, fields: List[Dict]) -> str:
        """Generate Pydantic schemas for validation"""
        type_mapping = {
            "text": "str",
            "email": "EmailStr",
            "password": "str",
            "number": "int",
            "float": "float",
            "date": "date",
            "datetime": "datetime",
            "textarea": "str",
            "boolean": "bool",
            "select": "str"
        }
        
        create_fields = []
        response_fields = []
        
        for f in fields:
            py_type = type_mapping.get(f.get("type", "text"), "str")
            
            if f.get("required"):
                create_fields.append(f'    {f["name"]}: {py_type}')
            else:
                create_fields.append(f'    {f["name"]}: Optional[{py_type}] = None')
            
            response_fields.append(f'    {f["name"]}: Optional[{py_type}] = None')
        
        return f'''"""
Pydantic schemas for {table}
Auto-generated by IntentForge
"""

from datetime import datetime, date
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, field_validator
import re


class {table.title().replace('_', '')}Base(BaseModel):
    """Base schema with common fields"""
{chr(10).join(create_fields)}


class {table.title().replace('_', '')}Create({table.title().replace('_', '')}Base):
    """Schema for creating new {table}"""
    
    @field_validator('*', mode='before')
    @classmethod
    def strip_strings(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v


class {table.title().replace('_', '')}Update(BaseModel):
    """Schema for updating {table} (all fields optional)"""
{chr(10).join(f.replace(': str', ': Optional[str] = None').replace(': int', ': Optional[int] = None').replace(': float', ': Optional[float] = None') for f in create_fields)}


class {table.title().replace('_', '')}Response({table.title().replace('_', '')}Base):
    """Schema for API responses"""
    id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class {table.title().replace('_', '')}List(BaseModel):
    """Schema for paginated list response"""
    items: list[{table.title().replace('_', '')}Response]
    total: int
    page: int
    per_page: int
    pages: int
'''
    
    @staticmethod
    def _generate_migration(table: str, fields: List[Dict]) -> str:
        """Generate Alembic migration"""
        type_mapping = {
            "text": "sa.String(255)",
            "email": "sa.String(255)",
            "password": "sa.String(255)",
            "number": "sa.Integer()",
            "float": "sa.Float()",
            "date": "sa.Date()",
            "datetime": "sa.DateTime(timezone=True)",
            "textarea": "sa.Text()",
            "boolean": "sa.Boolean()",
            "select": "sa.String(100)"
        }
        
        columns = []
        for f in fields:
            col_type = type_mapping.get(f.get("type", "text"), "sa.String(255)")
            nullable = "False" if f.get("required") else "True"
            columns.append(f"        sa.Column('{f['name']}', {col_type}, nullable={nullable}),")
        
        return f'''"""
Create {table} table

Auto-generated by IntentForge
"""

from alembic import op
import sqlalchemy as sa
from datetime import datetime

# revision identifiers
revision = 'auto_generated'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        '{table}',
        sa.Column('id', sa.Integer(), primary_key=True),
{chr(10).join(columns)}
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )
    
    # Add indexes
    op.create_index('idx_{table}_created_at', '{table}', ['created_at'])


def downgrade():
    op.drop_table('{table}')
'''
    
    @staticmethod
    def _generate_tests(table: str, fields: List[Dict]) -> str:
        """Generate pytest tests"""
        sample_data = {}
        for f in fields:
            if f.get("type") == "email":
                sample_data[f["name"]] = "test@example.com"
            elif f.get("type") == "number":
                sample_data[f["name"]] = 42
            elif f.get("type") == "boolean":
                sample_data[f["name"]] = True
            else:
                sample_data[f["name"]] = f"Test {f['name']}"
        
        return f'''"""
Tests for {table} API
Auto-generated by IntentForge
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.main import app
from app.models import {table.title().replace('_', '')}


SAMPLE_DATA = {json.dumps(sample_data, indent=4)}


@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def sample_{table.rstrip('s')}(session: AsyncSession):
    item = {table.title().replace('_', '')}(**SAMPLE_DATA)
    session.add(item)
    await session.commit()
    await session.refresh(item)
    return item


class Test{table.title().replace('_', '')}API:
    
    async def test_create_{table.rstrip('s')}(self, client: AsyncClient):
        response = await client.post("/api/{table}/", json=SAMPLE_DATA)
        assert response.status_code == 201
        data = response.json()
        assert data["id"] is not None
{chr(10).join(f'        assert data["{f["name"]}"] == SAMPLE_DATA["{f["name"]}"]' for f in fields)}
    
    async def test_get_{table}(self, client: AsyncClient, sample_{table.rstrip('s')}):
        response = await client.get("/api/{table}/")
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
    
    async def test_get_{table.rstrip('s')}_by_id(self, client: AsyncClient, sample_{table.rstrip('s')}):
        response = await client.get(f"/api/{table}/{{sample_{table.rstrip('s')}.id}}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == sample_{table.rstrip('s')}.id
    
    async def test_get_{table.rstrip('s')}_not_found(self, client: AsyncClient):
        response = await client.get("/api/{table}/99999")
        assert response.status_code == 404
    
    async def test_update_{table.rstrip('s')}(self, client: AsyncClient, sample_{table.rstrip('s')}):
        update_data = {{"name": "Updated Name"}}
        response = await client.put(
            f"/api/{table}/{{sample_{table.rstrip('s')}.id}}", 
            json=update_data
        )
        assert response.status_code == 200
    
    async def test_delete_{table.rstrip('s')}(self, client: AsyncClient, sample_{table.rstrip('s')}):
        response = await client.delete(f"/api/{table}/{{sample_{table.rstrip('s')}.id}}")
        assert response.status_code == 204
        
        # Verify deleted
        response = await client.get(f"/api/{table}/{{sample_{table.rstrip('s')}.id}}")
        assert response.status_code == 404
'''
    
    @staticmethod
    def crud_api(config: PatternConfig) -> Dict[str, str]:
        """Generate complete CRUD API pattern"""
        return FullstackPatterns.form_to_database(config)
    
    @staticmethod  
    def mqtt_handler(table: str) -> str:
        """Generate MQTT message handler for form submissions"""
        return f'''"""
MQTT Handler for {table}
Processes form submissions from any client via MQTT
Auto-generated by IntentForge
"""

import json
import asyncio
import logging
from typing import Dict, Any, Callable
import paho.mqtt.client as mqtt

from .database import async_session
from .models import {table.title().replace('_', '')}
from .schemas import {table.title().replace('_', '')}Create

logger = logging.getLogger(__name__)


class {table.title().replace('_', '')}MQTTHandler:
    """Handle MQTT messages for {table} operations"""
    
    TOPIC_BASE = "intentforge/forms/{table}"
    
    def __init__(
        self, 
        host: str = "localhost",
        port: int = 1883,
        client_id: str = "{table}-handler"
    ):
        self.host = host
        self.port = port
        self.client = mqtt.Client(client_id=client_id)
        
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        
        self._handlers: Dict[str, Callable] = {{
            "create": self._handle_create,
            "update": self._handle_update,
            "delete": self._handle_delete,
            "list": self._handle_list
        }}
    
    def _on_connect(self, client, userdata, flags, rc):
        logger.info(f"Connected to MQTT broker with code {{rc}}")
        client.subscribe(f"{{self.TOPIC_BASE}}/#")
    
    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            action = payload.get("action", "create")
            request_id = payload.get("request_id")
            
            handler = self._handlers.get(action)
            if handler:
                asyncio.create_task(
                    self._process_message(handler, payload, request_id)
                )
            else:
                self._send_error(request_id, f"Unknown action: {{action}}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {{e}}")
    
    async def _process_message(self, handler, payload, request_id):
        try:
            result = await handler(payload.get("data", {{}}))
            self._send_response(request_id, result)
        except Exception as e:
            logger.error(f"Handler error: {{e}}")
            self._send_error(request_id, str(e))
    
    async def _handle_create(self, data: Dict[str, Any]) -> Dict:
        async with async_session() as session:
            validated = {table.title().replace('_', '')}Create(**data)
            item = {table.title().replace('_', '')}(**validated.model_dump())
            session.add(item)
            await session.commit()
            await session.refresh(item)
            return item.to_dict()
    
    async def _handle_update(self, data: Dict[str, Any]) -> Dict:
        item_id = data.pop("id", None)
        if not item_id:
            raise ValueError("ID required for update")
        
        async with async_session() as session:
            from sqlalchemy import select
            result = await session.execute(
                select({table.title().replace('_', '')}).where({table.title().replace('_', '')}.id == item_id)
            )
            item = result.scalar_one_or_none()
            
            if not item:
                raise ValueError(f"{table} not found")
            
            for key, value in data.items():
                if hasattr(item, key):
                    setattr(item, key, value)
            
            await session.commit()
            await session.refresh(item)
            return item.to_dict()
    
    async def _handle_delete(self, data: Dict[str, Any]) -> Dict:
        item_id = data.get("id")
        if not item_id:
            raise ValueError("ID required for delete")
        
        async with async_session() as session:
            from sqlalchemy import select
            result = await session.execute(
                select({table.title().replace('_', '')}).where({table.title().replace('_', '')}.id == item_id)
            )
            item = result.scalar_one_or_none()
            
            if not item:
                raise ValueError(f"{table} not found")
            
            await session.delete(item)
            await session.commit()
            return {{"deleted": True, "id": item_id}}
    
    async def _handle_list(self, data: Dict[str, Any]) -> Dict:
        async with async_session() as session:
            from sqlalchemy import select
            
            limit = data.get("limit", 50)
            offset = data.get("offset", 0)
            
            result = await session.execute(
                select({table.title().replace('_', '')}).limit(limit).offset(offset)
            )
            items = result.scalars().all()
            
            return {{
                "items": [i.to_dict() for i in items],
                "count": len(items)
            }}
    
    def _send_response(self, request_id: str, data: Dict):
        if request_id:
            self.client.publish(
                f"{{self.TOPIC_BASE}}/response/{{request_id}}",
                json.dumps({{"success": True, "data": data}})
            )
    
    def _send_error(self, request_id: str, error: str):
        if request_id:
            self.client.publish(
                f"{{self.TOPIC_BASE}}/response/{{request_id}}",
                json.dumps({{"success": False, "error": error}})
            )
    
    def start(self):
        self.client.connect(self.host, self.port)
        self.client.loop_start()
    
    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()
'''

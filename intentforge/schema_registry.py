"""
Schema Registry - JSON Schema validation and DSL type definitions
Ensures correct data formats before code generation
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple, Type
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib

from pydantic import BaseModel, Field, field_validator, ValidationError
from jsonschema import validate, ValidationError as JSONSchemaError, Draft7Validator


class SchemaType(Enum):
    """Supported schema types"""
    INTENT_REQUEST = "intent_request"
    FORM_DATA = "form_data"
    DOM_SELECTOR = "dom_selector"
    SQL_QUERY = "sql_query"
    API_ENDPOINT = "api_endpoint"
    DATABASE_CONFIG = "database_config"
    MQTT_MESSAGE = "mqtt_message"


# ============================================================================
# JSON Schemas for validation
# ============================================================================

SCHEMAS: Dict[SchemaType, Dict[str, Any]] = {
    SchemaType.INTENT_REQUEST: {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["description"],
        "properties": {
            "description": {
                "type": "string",
                "minLength": 3,
                "maxLength": 1000,
                "description": "Natural language intent description"
            },
            "intent_type": {
                "type": "string",
                "enum": ["api_endpoint", "database_schema", "firmware_function", 
                        "event_handler", "workflow", "validation_rule", 
                        "ui_component", "mqtt_handler"]
            },
            "target_platform": {
                "type": "string",
                "enum": ["python_fastapi", "python_flask", "nodejs_express",
                        "arduino_cpp", "esp32_micropython", "jetson_python",
                        "raspberry_pi", "generic_python"]
            },
            "context": {
                "type": "object",
                "additionalProperties": True
            },
            "constraints": {
                "type": "array",
                "items": {"type": "string"}
            },
            "priority": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10
            }
        }
    },
    
    SchemaType.FORM_DATA: {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["form_id", "fields"],
        "properties": {
            "form_id": {
                "type": "string",
                "pattern": "^[a-zA-Z][a-zA-Z0-9_-]*$"
            },
            "action": {
                "type": "string",
                "format": "uri-reference"
            },
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"]
            },
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "type"],
                    "properties": {
                        "name": {"type": "string"},
                        "type": {
                            "type": "string",
                            "enum": ["text", "email", "password", "number", 
                                    "date", "datetime", "select", "checkbox",
                                    "radio", "textarea", "file", "hidden"]
                        },
                        "required": {"type": "boolean"},
                        "validation": {
                            "type": "object",
                            "properties": {
                                "pattern": {"type": "string"},
                                "min": {"type": "number"},
                                "max": {"type": "number"},
                                "minLength": {"type": "integer"},
                                "maxLength": {"type": "integer"}
                            }
                        }
                    }
                }
            },
            "database": {
                "type": "object",
                "properties": {
                    "table": {"type": "string"},
                    "operation": {
                        "type": "string",
                        "enum": ["insert", "update", "upsert"]
                    }
                }
            }
        }
    },
    
    SchemaType.DOM_SELECTOR: {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["selector"],
        "properties": {
            "selector": {
                "type": "string",
                "description": "CSS selector or XPath"
            },
            "selector_type": {
                "type": "string",
                "enum": ["css", "xpath", "id", "class", "name", "data-attr"]
            },
            "action": {
                "type": "string",
                "enum": ["get_value", "set_value", "get_text", "set_text",
                        "get_attribute", "set_attribute", "remove",
                        "add_class", "remove_class", "toggle_class",
                        "click", "focus", "blur", "submit"]
            },
            "value": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "number"},
                    {"type": "boolean"},
                    {"type": "null"}
                ]
            },
            "attribute": {"type": "string"},
            "event_handler": {
                "type": "object",
                "properties": {
                    "event": {"type": "string"},
                    "callback": {"type": "string"}
                }
            }
        }
    },
    
    SchemaType.SQL_QUERY: {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["operation"],
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["SELECT", "INSERT", "UPDATE", "DELETE", 
                        "CREATE", "ALTER", "DROP", "TRUNCATE"]
            },
            "table": {
                "type": "string",
                "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*$"
            },
            "columns": {
                "type": "array",
                "items": {"type": "string"}
            },
            "conditions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "column": {"type": "string"},
                        "operator": {
                            "type": "string",
                            "enum": ["=", "!=", "<", ">", "<=", ">=", 
                                    "LIKE", "IN", "NOT IN", "IS NULL", 
                                    "IS NOT NULL", "BETWEEN"]
                        },
                        "value": {},
                        "parameterized": {"type": "boolean", "default": True}
                    }
                }
            },
            "joins": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["INNER", "LEFT", "RIGHT", "FULL", "CROSS"]
                        },
                        "table": {"type": "string"},
                        "on": {"type": "string"}
                    }
                }
            },
            "order_by": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "column": {"type": "string"},
                        "direction": {
                            "type": "string",
                            "enum": ["ASC", "DESC"]
                        }
                    }
                }
            },
            "limit": {"type": "integer", "minimum": 1},
            "offset": {"type": "integer", "minimum": 0}
        }
    },
    
    SchemaType.DATABASE_CONFIG: {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["driver"],
        "properties": {
            "driver": {
                "type": "string",
                "enum": ["postgresql", "mysql", "sqlite", "mssql", "oracle"]
            },
            "host": {"type": "string"},
            "port": {"type": "integer"},
            "database": {"type": "string"},
            "username": {"type": "string"},
            "password": {"type": "string"},
            "ssl": {"type": "boolean"},
            "pool_size": {"type": "integer", "minimum": 1, "maximum": 100},
            "env_vars": {
                "type": "object",
                "description": "Mapping of config keys to environment variables",
                "properties": {
                    "host": {"type": "string"},
                    "port": {"type": "string"},
                    "database": {"type": "string"},
                    "username": {"type": "string"},
                    "password": {"type": "string"}
                }
            }
        }
    },
    
    SchemaType.MQTT_MESSAGE: {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["topic", "payload"],
        "properties": {
            "topic": {
                "type": "string",
                "pattern": "^[a-zA-Z0-9/_#+]+$"
            },
            "payload": {
                "oneOf": [
                    {"type": "object"},
                    {"type": "string"},
                    {"type": "array"}
                ]
            },
            "qos": {
                "type": "integer",
                "enum": [0, 1, 2]
            },
            "retain": {"type": "boolean"}
        }
    }
}


# ============================================================================
# Pydantic Models for typed validation
# ============================================================================

class FormField(BaseModel):
    """Form field definition"""
    name: str = Field(..., min_length=1, max_length=100)
    field_type: str = Field(..., alias="type")
    label: Optional[str] = None
    required: bool = False
    default: Optional[Any] = None
    validation: Optional[Dict[str, Any]] = None
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', v):
            raise ValueError('Field name must be a valid identifier')
        return v


class FormDefinition(BaseModel):
    """Complete form definition for DOM/Database integration"""
    form_id: str = Field(..., pattern=r'^[a-zA-Z][a-zA-Z0-9_-]*$')
    action: str = "/api/submit"
    method: str = "POST"
    fields: List[FormField]
    
    # Database integration
    target_table: Optional[str] = None
    on_success: Optional[str] = None
    on_error: Optional[str] = None
    
    def to_dom_dsl(self) -> Dict[str, Any]:
        """Convert to DOM DSL for frontend generation"""
        return {
            "type": "form",
            "id": self.form_id,
            "action": self.action,
            "method": self.method,
            "elements": [
                {
                    "tag": "input" if f.field_type != "textarea" else "textarea",
                    "type": f.field_type if f.field_type != "textarea" else None,
                    "name": f.name,
                    "id": f"{self.form_id}_{f.name}",
                    "required": f.required,
                    "label": f.label or f.name.replace('_', ' ').title()
                }
                for f in self.fields
            ],
            "submit_button": {
                "tag": "button",
                "type": "submit",
                "text": "Submit"
            }
        }
    
    def to_sql_insert(self) -> Dict[str, Any]:
        """Convert to SQL DSL for database insertion"""
        return {
            "operation": "INSERT",
            "table": self.target_table or self.form_id,
            "columns": [f.name for f in self.fields],
            "parameterized": True,
            "returning": ["id", "created_at"]
        }


class DOMOperation(BaseModel):
    """DOM manipulation operation"""
    selector: str
    selector_type: str = "css"
    action: str
    value: Optional[Any] = None
    attribute: Optional[str] = None


class SQLOperation(BaseModel):
    """SQL operation definition with parameterization"""
    operation: str
    table: str = Field(..., pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*$')
    columns: List[str] = []
    values: Optional[Dict[str, Any]] = None
    conditions: List[Dict[str, Any]] = []
    
    def to_parameterized_query(self) -> Tuple[str, Dict[str, Any]]:
        """Generate parameterized SQL query"""
        params = {}
        
        if self.operation.upper() == "SELECT":
            cols = ", ".join(self.columns) if self.columns else "*"
            query = f"SELECT {cols} FROM {self.table}"
            
            if self.conditions:
                where_parts = []
                for i, cond in enumerate(self.conditions):
                    param_name = f"p{i}"
                    where_parts.append(f"{cond['column']} {cond.get('operator', '=')} %({param_name})s")
                    params[param_name] = cond.get('value')
                query += " WHERE " + " AND ".join(where_parts)
            
            return query, params
        
        elif self.operation.upper() == "INSERT":
            cols = ", ".join(self.columns)
            placeholders = ", ".join(f"%({c})s" for c in self.columns)
            query = f"INSERT INTO {self.table} ({cols}) VALUES ({placeholders}) RETURNING *"
            params = self.values or {}
            return query, params
        
        elif self.operation.upper() == "UPDATE":
            set_parts = [f"{c} = %({c})s" for c in self.columns]
            query = f"UPDATE {self.table} SET {', '.join(set_parts)}"
            params = self.values or {}
            
            if self.conditions:
                where_parts = []
                for i, cond in enumerate(self.conditions):
                    param_name = f"where_{i}"
                    where_parts.append(f"{cond['column']} {cond.get('operator', '=')} %({param_name})s")
                    params[param_name] = cond.get('value')
                query += " WHERE " + " AND ".join(where_parts)
            
            query += " RETURNING *"
            return query, params
        
        elif self.operation.upper() == "DELETE":
            query = f"DELETE FROM {self.table}"
            
            if self.conditions:
                where_parts = []
                for i, cond in enumerate(self.conditions):
                    param_name = f"p{i}"
                    where_parts.append(f"{cond['column']} {cond.get('operator', '=')} %({param_name})s")
                    params[param_name] = cond.get('value')
                query += " WHERE " + " AND ".join(where_parts)
            else:
                raise ValueError("DELETE without conditions is not allowed")
            
            query += " RETURNING *"
            return query, params
        
        raise ValueError(f"Unsupported operation: {self.operation}")


# ============================================================================
# Schema Registry
# ============================================================================

@dataclass
class ValidationResult:
    """Result of schema validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    normalized_data: Optional[Dict[str, Any]] = None


class SchemaRegistry:
    """
    Central registry for all schemas
    Handles validation, caching, and schema versioning
    """
    
    def __init__(self):
        self._schemas = SCHEMAS.copy()
        self._validators: Dict[SchemaType, Draft7Validator] = {}
        self._custom_schemas: Dict[str, Dict[str, Any]] = {}
        self._validation_cache: Dict[str, ValidationResult] = {}
        
        # Initialize validators
        for schema_type, schema in self._schemas.items():
            self._validators[schema_type] = Draft7Validator(schema)
    
    def register_schema(
        self, 
        name: str, 
        schema: Dict[str, Any],
        schema_type: Optional[SchemaType] = None
    ) -> None:
        """Register a custom schema"""
        self._custom_schemas[name] = schema
        if schema_type:
            self._schemas[schema_type] = schema
            self._validators[schema_type] = Draft7Validator(schema)
    
    def validate(
        self,
        data: Dict[str, Any],
        schema_type: SchemaType,
        use_cache: bool = True
    ) -> ValidationResult:
        """
        Validate data against schema
        Uses caching for repeated validations
        """
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(data, schema_type)
            if cache_key in self._validation_cache:
                return self._validation_cache[cache_key]
        
        errors = []
        warnings = []
        
        # JSON Schema validation
        validator = self._validators.get(schema_type)
        if validator:
            for error in validator.iter_errors(data):
                errors.append(f"{error.path}: {error.message}" if error.path else error.message)
        
        # Additional semantic validation
        semantic_errors = self._semantic_validation(data, schema_type)
        errors.extend(semantic_errors)
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            normalized_data=data if len(errors) == 0 else None
        )
        
        # Cache result
        if use_cache:
            self._validation_cache[cache_key] = result
        
        return result
    
    def _get_cache_key(self, data: Dict[str, Any], schema_type: SchemaType) -> str:
        """Generate cache key from data and schema type"""
        content = json.dumps(data, sort_keys=True) + schema_type.value
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _semantic_validation(
        self, 
        data: Dict[str, Any], 
        schema_type: SchemaType
    ) -> List[str]:
        """Additional semantic validation beyond JSON Schema"""
        errors = []
        
        if schema_type == SchemaType.SQL_QUERY:
            # Check for dangerous patterns
            if data.get('operation') in ('UPDATE', 'DELETE'):
                if not data.get('conditions'):
                    errors.append("UPDATE/DELETE operations require conditions (WHERE clause)")
        
        if schema_type == SchemaType.FORM_DATA:
            # Validate field names uniqueness
            field_names = [f.get('name') for f in data.get('fields', [])]
            if len(field_names) != len(set(field_names)):
                errors.append("Duplicate field names detected")
        
        if schema_type == SchemaType.DATABASE_CONFIG:
            # Check for direct credentials (should use env vars)
            if data.get('password') and not data.get('env_vars', {}).get('password'):
                errors.append("Database password should reference an environment variable")
        
        return errors
    
    def validate_form_to_sql(self, form_data: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate form data and generate safe SQL
        Returns: (is_valid, sql_query, parameters)
        """
        # Validate form data
        form_result = self.validate(form_data, SchemaType.FORM_DATA)
        if not form_result.is_valid:
            return False, "", {"errors": form_result.errors}
        
        try:
            form = FormDefinition(**form_data)
            sql_def = form.to_sql_insert()
            
            # Generate parameterized SQL
            operation = SQLOperation(
                operation=sql_def["operation"],
                table=sql_def["table"],
                columns=sql_def["columns"]
            )
            
            query, params = operation.to_parameterized_query()
            return True, query, {"template": params, "columns": sql_def["columns"]}
            
        except ValidationError as e:
            return False, "", {"errors": [str(e)]}
    
    def clear_cache(self) -> None:
        """Clear validation cache"""
        self._validation_cache.clear()


# Global registry instance
_registry: Optional[SchemaRegistry] = None


def get_registry() -> SchemaRegistry:
    """Get global schema registry instance"""
    global _registry
    if _registry is None:
        _registry = SchemaRegistry()
    return _registry

"""
Code Validator - Multi-level validation pipeline
Level 1: Syntax validation (AST parsing)
Level 2: Security analysis (dangerous patterns, injections)
Level 3: Semantic validation (type checking, logic verification)
"""

import ast
import re
import hashlib
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

from .config import get_settings, ValidationSettings
from .core import TargetPlatform

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels"""
    SYNTAX = "syntax"
    SECURITY = "security"
    SEMANTIC = "semantic"
    FULL = "full"


class SecurityRisk(Enum):
    """Security risk levels"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ValidationError:
    """Detailed validation error"""
    level: ValidationLevel
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    code_snippet: Optional[str] = None
    suggestion: Optional[str] = None
    risk_level: SecurityRisk = SecurityRisk.NONE
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "message": self.message,
            "line": self.line,
            "column": self.column,
            "code_snippet": self.code_snippet,
            "suggestion": self.suggestion,
            "risk_level": self.risk_level.value
        }


@dataclass
class ValidationResult:
    """Complete validation result"""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    security_score: float = 100.0  # 0-100, higher is safer
    complexity_score: float = 0.0  # cyclomatic complexity
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error: ValidationError) -> None:
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: ValidationError) -> None:
        self.warnings.append(warning)
    
    @property
    def error_messages(self) -> List[str]:
        return [e.message for e in self.errors]


class BaseValidator(ABC):
    """Base class for validators"""
    
    @abstractmethod
    def validate(self, code: str, language: str) -> ValidationResult:
        pass


class SyntaxValidator(BaseValidator):
    """
    Level 1: Syntax Validation
    Parses code to ensure it's syntactically correct
    """
    
    def validate(self, code: str, language: str) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        validators = {
            "python": self._validate_python,
            "javascript": self._validate_javascript,
            "sql": self._validate_sql,
            "html": self._validate_html,
            "json": self._validate_json,
        }
        
        validator = validators.get(language.lower())
        if validator:
            validator(code, result)
        else:
            result.add_warning(ValidationError(
                level=ValidationLevel.SYNTAX,
                message=f"No syntax validator available for {language}"
            ))
        
        return result
    
    def _validate_python(self, code: str, result: ValidationResult) -> None:
        """Validate Python syntax using AST"""
        try:
            ast.parse(code)
            result.metadata["ast_valid"] = True
        except SyntaxError as e:
            result.add_error(ValidationError(
                level=ValidationLevel.SYNTAX,
                message=f"Python syntax error: {e.msg}",
                line=e.lineno,
                column=e.offset,
                code_snippet=e.text,
                suggestion="Check for missing colons, brackets, or indentation"
            ))
    
    def _validate_javascript(self, code: str, result: ValidationResult) -> None:
        """Basic JavaScript syntax validation"""
        # Check balanced brackets
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        
        for i, char in enumerate(code):
            if char in brackets:
                stack.append((char, i))
            elif char in brackets.values():
                if not stack:
                    result.add_error(ValidationError(
                        level=ValidationLevel.SYNTAX,
                        message=f"Unmatched closing bracket '{char}'",
                        column=i
                    ))
                else:
                    open_bracket, _ = stack.pop()
                    if brackets[open_bracket] != char:
                        result.add_error(ValidationError(
                            level=ValidationLevel.SYNTAX,
                            message=f"Mismatched brackets: '{open_bracket}' and '{char}'"
                        ))
        
        for bracket, pos in stack:
            result.add_error(ValidationError(
                level=ValidationLevel.SYNTAX,
                message=f"Unclosed bracket '{bracket}'",
                column=pos
            ))
    
    def _validate_sql(self, code: str, result: ValidationResult) -> None:
        """Validate SQL syntax"""
        # Basic SQL keyword validation
        required_patterns = [
            (r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b', 
             "SQL statement must start with a valid keyword")
        ]
        
        code_upper = code.upper()
        for pattern, message in required_patterns:
            if not re.search(pattern, code_upper):
                result.add_warning(ValidationError(
                    level=ValidationLevel.SYNTAX,
                    message=message
                ))
    
    def _validate_html(self, code: str, result: ValidationResult) -> None:
        """Validate HTML structure"""
        from html.parser import HTMLParser
        
        class HTMLValidator(HTMLParser):
            def __init__(self):
                super().__init__()
                self.tag_stack = []
                self.errors = []
            
            def handle_starttag(self, tag, attrs):
                void_tags = {'br', 'hr', 'img', 'input', 'meta', 'link', 'area', 'base', 'col'}
                if tag.lower() not in void_tags:
                    self.tag_stack.append(tag)
            
            def handle_endtag(self, tag):
                if self.tag_stack and self.tag_stack[-1] == tag:
                    self.tag_stack.pop()
                elif tag in self.tag_stack:
                    self.errors.append(f"Mismatched tag: </{tag}>")
        
        try:
            validator = HTMLValidator()
            validator.feed(code)
            for error in validator.errors:
                result.add_error(ValidationError(
                    level=ValidationLevel.SYNTAX,
                    message=error
                ))
            for tag in validator.tag_stack:
                result.add_warning(ValidationError(
                    level=ValidationLevel.SYNTAX,
                    message=f"Unclosed tag: <{tag}>"
                ))
        except Exception as e:
            result.add_error(ValidationError(
                level=ValidationLevel.SYNTAX,
                message=f"HTML parsing error: {str(e)}"
            ))
    
    def _validate_json(self, code: str, result: ValidationResult) -> None:
        """Validate JSON syntax"""
        import json
        try:
            json.loads(code)
        except json.JSONDecodeError as e:
            result.add_error(ValidationError(
                level=ValidationLevel.SYNTAX,
                message=f"JSON syntax error: {e.msg}",
                line=e.lineno,
                column=e.colno
            ))


class SecurityValidator(BaseValidator):
    """
    Level 2: Security Validation
    Detects dangerous patterns, injections, and security risks
    """
    
    # Dangerous patterns by language
    DANGEROUS_PATTERNS = {
        "python": [
            (r'\beval\s*\(', SecurityRisk.CRITICAL, "eval() can execute arbitrary code"),
            (r'\bexec\s*\(', SecurityRisk.CRITICAL, "exec() can execute arbitrary code"),
            (r'\b__import__\s*\(', SecurityRisk.HIGH, "Dynamic imports can be dangerous"),
            (r'\bcompile\s*\(', SecurityRisk.HIGH, "compile() can create executable code"),
            (r'os\.system\s*\(', SecurityRisk.CRITICAL, "os.system() executes shell commands"),
            (r'subprocess\.(call|run|Popen)\s*\(.*shell\s*=\s*True', SecurityRisk.CRITICAL, 
             "Shell execution with shell=True is dangerous"),
            (r'pickle\.loads?\s*\(', SecurityRisk.HIGH, "pickle can execute arbitrary code"),
            (r'open\s*\([^)]*["\'][^"\']*\.\.[^"\']*["\']', SecurityRisk.MEDIUM, 
             "Path traversal detected"),
        ],
        "javascript": [
            (r'\beval\s*\(', SecurityRisk.CRITICAL, "eval() executes arbitrary code"),
            (r'new\s+Function\s*\(', SecurityRisk.CRITICAL, "Function constructor is like eval"),
            (r'innerHTML\s*=', SecurityRisk.MEDIUM, "innerHTML can lead to XSS"),
            (r'document\.write\s*\(', SecurityRisk.MEDIUM, "document.write can be dangerous"),
            (r'\.insertAdjacentHTML\s*\(', SecurityRisk.MEDIUM, "Can lead to XSS"),
        ],
        "sql": [
            (r';\s*DROP\s+', SecurityRisk.CRITICAL, "SQL injection: DROP statement"),
            (r';\s*DELETE\s+FROM\s+\w+\s*;', SecurityRisk.CRITICAL, "Unrestricted DELETE"),
            (r'UNION\s+SELECT', SecurityRisk.HIGH, "Potential SQL injection via UNION"),
            (r'--\s*$', SecurityRisk.MEDIUM, "SQL comment at end of line"),
            (r"'\s*OR\s+'1'\s*=\s*'1", SecurityRisk.CRITICAL, "Classic SQL injection pattern"),
            (r';\s*UPDATE\s+\w+\s+SET.*WHERE\s+1\s*=\s*1', SecurityRisk.CRITICAL, 
             "Unrestricted UPDATE"),
        ],
        "html": [
            (r'<script[^>]*>.*?</script>', SecurityRisk.MEDIUM, "Inline script detected"),
            (r'javascript:', SecurityRisk.HIGH, "JavaScript protocol in URL"),
            (r'on\w+\s*=', SecurityRisk.MEDIUM, "Inline event handler"),
            (r'<iframe', SecurityRisk.LOW, "iframes can be security risk"),
        ]
    }
    
    # Forbidden imports/modules
    FORBIDDEN_IMPORTS = {
        "python": {
            "os.system", "subprocess.call", "subprocess.run", 
            "pickle", "marshal", "shelve",
            "ctypes", "importlib.__import__"
        }
    }
    
    def validate(self, code: str, language: str) -> ValidationResult:
        result = ValidationResult(is_valid=True, security_score=100.0)
        
        # Check dangerous patterns
        patterns = self.DANGEROUS_PATTERNS.get(language.lower(), [])
        for pattern, risk, message in patterns:
            matches = list(re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE))
            for match in matches:
                # Calculate line number
                line_num = code[:match.start()].count('\n') + 1
                
                error = ValidationError(
                    level=ValidationLevel.SECURITY,
                    message=message,
                    line=line_num,
                    code_snippet=match.group()[:50],
                    risk_level=risk,
                    suggestion=self._get_suggestion(pattern, language)
                )
                
                if risk in (SecurityRisk.CRITICAL, SecurityRisk.HIGH):
                    result.add_error(error)
                    result.security_score -= 25
                else:
                    result.add_warning(error)
                    result.security_score -= 10
        
        # Python-specific: Check AST for dangerous calls
        if language.lower() == "python":
            self._validate_python_ast(code, result)
        
        # Ensure score doesn't go negative
        result.security_score = max(0, result.security_score)
        
        return result
    
    def _validate_python_ast(self, code: str, result: ValidationResult) -> None:
        """Deep AST analysis for Python"""
        try:
            tree = ast.parse(code)
            
            class SecurityVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.issues = []
                
                def visit_Call(self, node):
                    # Check for dangerous function calls
                    func_name = self._get_func_name(node.func)
                    
                    dangerous_funcs = {
                        'eval': SecurityRisk.CRITICAL,
                        'exec': SecurityRisk.CRITICAL,
                        '__import__': SecurityRisk.HIGH,
                        'compile': SecurityRisk.HIGH,
                        'globals': SecurityRisk.MEDIUM,
                        'locals': SecurityRisk.MEDIUM,
                        'setattr': SecurityRisk.MEDIUM,
                        'delattr': SecurityRisk.MEDIUM,
                    }
                    
                    if func_name in dangerous_funcs:
                        self.issues.append((
                            func_name, 
                            dangerous_funcs[func_name],
                            node.lineno
                        ))
                    
                    self.generic_visit(node)
                
                def visit_Import(self, node):
                    for alias in node.names:
                        if alias.name in ('pickle', 'marshal', 'ctypes'):
                            self.issues.append((
                                f"import {alias.name}",
                                SecurityRisk.HIGH,
                                node.lineno
                            ))
                    self.generic_visit(node)
                
                def _get_func_name(self, node) -> str:
                    if isinstance(node, ast.Name):
                        return node.id
                    elif isinstance(node, ast.Attribute):
                        return f"{self._get_func_name(node.value)}.{node.attr}"
                    return ""
            
            visitor = SecurityVisitor()
            visitor.visit(tree)
            
            for func_name, risk, line in visitor.issues:
                error = ValidationError(
                    level=ValidationLevel.SECURITY,
                    message=f"Dangerous function detected: {func_name}",
                    line=line,
                    risk_level=risk
                )
                if risk in (SecurityRisk.CRITICAL, SecurityRisk.HIGH):
                    result.add_error(error)
                else:
                    result.add_warning(error)
                    
        except SyntaxError:
            pass  # Syntax errors handled by SyntaxValidator
    
    def _get_suggestion(self, pattern: str, language: str) -> str:
        """Get security fix suggestion"""
        suggestions = {
            r'\beval\s*\(': "Use ast.literal_eval() for safe evaluation or JSON parsing",
            r'\bexec\s*\(': "Avoid dynamic code execution; use configuration or plugins instead",
            r'os\.system': "Use subprocess.run() with shell=False and explicit arguments",
            r'innerHTML': "Use textContent or createElement/appendChild for DOM manipulation",
            r'UNION\s+SELECT': "Use parameterized queries to prevent SQL injection",
        }
        
        for pat, suggestion in suggestions.items():
            if pat in pattern:
                return suggestion
        return "Review this code for potential security issues"


class SemanticValidator(BaseValidator):
    """
    Level 3: Semantic Validation
    Validates logic, types, and business rules
    """
    
    def validate(self, code: str, language: str) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        if language.lower() == "python":
            self._validate_python_semantics(code, result)
        elif language.lower() in ("javascript", "typescript"):
            self._validate_js_semantics(code, result)
        elif language.lower() == "sql":
            self._validate_sql_semantics(code, result)
        
        # Calculate complexity
        result.complexity_score = self._calculate_complexity(code, language)
        
        return result
    
    def _validate_python_semantics(self, code: str, result: ValidationResult) -> None:
        """Validate Python semantic correctness"""
        try:
            tree = ast.parse(code)
            
            # Check for undefined variables
            defined_names: Set[str] = set()
            used_names: Set[str] = set()
            
            class NameCollector(ast.NodeVisitor):
                def visit_FunctionDef(self, node):
                    defined_names.add(node.name)
                    for arg in node.args.args:
                        defined_names.add(arg.arg)
                    self.generic_visit(node)
                
                def visit_Assign(self, node):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            defined_names.add(target.id)
                    self.generic_visit(node)
                
                def visit_Name(self, node):
                    if isinstance(node.ctx, ast.Load):
                        used_names.add(node.id)
                    self.generic_visit(node)
            
            collector = NameCollector()
            collector.visit(tree)
            
            # Built-in names that are always available
            builtins = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))
            common_imports = {'os', 'sys', 'json', 'typing', 'dataclass', 'Optional', 'List', 'Dict', 'Any'}
            
            undefined = used_names - defined_names - builtins - common_imports
            for name in undefined:
                result.add_warning(ValidationError(
                    level=ValidationLevel.SEMANTIC,
                    message=f"Potentially undefined name: '{name}'",
                    suggestion=f"Ensure '{name}' is imported or defined"
                ))
            
            # Check for unreachable code
            self._check_unreachable_code(tree, result)
            
        except SyntaxError:
            pass
    
    def _check_unreachable_code(self, tree: ast.AST, result: ValidationResult) -> None:
        """Detect unreachable code after return/raise"""
        class UnreachableChecker(ast.NodeVisitor):
            def __init__(self):
                self.issues = []
            
            def visit_FunctionDef(self, node):
                for i, stmt in enumerate(node.body[:-1]):
                    if isinstance(stmt, (ast.Return, ast.Raise)):
                        next_stmt = node.body[i + 1]
                        if not isinstance(next_stmt, (ast.FunctionDef, ast.ClassDef)):
                            self.issues.append(next_stmt.lineno)
                self.generic_visit(node)
        
        checker = UnreachableChecker()
        checker.visit(tree)
        
        for line in checker.issues:
            result.add_warning(ValidationError(
                level=ValidationLevel.SEMANTIC,
                message="Unreachable code detected",
                line=line,
                suggestion="Remove code after return/raise statement"
            ))
    
    def _validate_js_semantics(self, code: str, result: ValidationResult) -> None:
        """Basic JavaScript semantic checks"""
        # Check for common issues
        patterns = [
            (r'==\s*null\b', "Use === for strict equality comparison"),
            (r'var\s+\w+', "Consider using 'let' or 'const' instead of 'var'"),
            (r'\.then\([^)]*\)\s*$', "Promise chain without .catch() for error handling"),
        ]
        
        for pattern, message in patterns:
            if re.search(pattern, code):
                result.add_warning(ValidationError(
                    level=ValidationLevel.SEMANTIC,
                    message=message
                ))
    
    def _validate_sql_semantics(self, code: str, result: ValidationResult) -> None:
        """Validate SQL semantic correctness"""
        # Check for SELECT *
        if re.search(r'SELECT\s+\*', code, re.IGNORECASE):
            result.add_warning(ValidationError(
                level=ValidationLevel.SEMANTIC,
                message="SELECT * is not recommended; specify columns explicitly",
                suggestion="List specific columns for better performance and clarity"
            ))
        
        # Check for missing WHERE in UPDATE/DELETE
        if re.search(r'\b(UPDATE|DELETE)\b', code, re.IGNORECASE):
            if not re.search(r'\bWHERE\b', code, re.IGNORECASE):
                result.add_error(ValidationError(
                    level=ValidationLevel.SEMANTIC,
                    message="UPDATE/DELETE without WHERE clause affects all rows",
                    suggestion="Add WHERE clause to limit affected rows",
                    risk_level=SecurityRisk.HIGH
                ))
    
    def _calculate_complexity(self, code: str, language: str) -> float:
        """Calculate cyclomatic complexity"""
        if language.lower() != "python":
            return 0.0
        
        try:
            tree = ast.parse(code)
            
            complexity = 1  # Base complexity
            
            class ComplexityVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.complexity = 0
                
                def visit_If(self, node):
                    self.complexity += 1
                    self.generic_visit(node)
                
                def visit_For(self, node):
                    self.complexity += 1
                    self.generic_visit(node)
                
                def visit_While(self, node):
                    self.complexity += 1
                    self.generic_visit(node)
                
                def visit_ExceptHandler(self, node):
                    self.complexity += 1
                    self.generic_visit(node)
                
                def visit_BoolOp(self, node):
                    self.complexity += len(node.values) - 1
                    self.generic_visit(node)
            
            visitor = ComplexityVisitor()
            visitor.visit(tree)
            
            return complexity + visitor.complexity
            
        except SyntaxError:
            return 0.0


class CodeValidator:
    """
    Main validator orchestrator
    Runs all validation levels and aggregates results
    """
    
    def __init__(self, sandbox_mode: bool = True, settings: Optional[ValidationSettings] = None):
        self.sandbox_mode = sandbox_mode
        self.settings = settings or get_settings().validation
        
        self.syntax_validator = SyntaxValidator()
        self.security_validator = SecurityValidator()
        self.semantic_validator = SemanticValidator()
    
    async def validate(
        self,
        code: str,
        language: str,
        target_platform: Optional[TargetPlatform] = None,
        level: ValidationLevel = ValidationLevel.FULL
    ) -> Tuple[bool, List[str]]:
        """
        Run validation pipeline
        Returns (is_valid, error_messages)
        """
        results: List[ValidationResult] = []
        
        # Check code length
        if len(code) > self.settings.max_code_length:
            return False, [f"Code exceeds maximum length of {self.settings.max_code_length} characters"]
        
        # Level 1: Syntax
        if level in (ValidationLevel.SYNTAX, ValidationLevel.FULL):
            if self.settings.enable_syntax_check:
                results.append(self.syntax_validator.validate(code, language))
        
        # Level 2: Security
        if level in (ValidationLevel.SECURITY, ValidationLevel.FULL):
            if self.settings.enable_security_check:
                results.append(self.security_validator.validate(code, language))
        
        # Level 3: Semantic
        if level in (ValidationLevel.SEMANTIC, ValidationLevel.FULL):
            if self.settings.enable_semantic_check:
                results.append(self.semantic_validator.validate(code, language))
        
        # Aggregate results
        all_errors = []
        is_valid = True
        
        for result in results:
            if not result.is_valid:
                is_valid = False
            all_errors.extend(result.error_messages)
        
        return is_valid, all_errors
    
    def validate_sync(
        self,
        code: str,
        language: str,
        level: ValidationLevel = ValidationLevel.FULL
    ) -> ValidationResult:
        """Synchronous validation for simple use cases"""
        combined = ValidationResult(is_valid=True)
        
        for validator in [self.syntax_validator, self.security_validator, self.semantic_validator]:
            result = validator.validate(code, language)
            combined.errors.extend(result.errors)
            combined.warnings.extend(result.warnings)
            if not result.is_valid:
                combined.is_valid = False
        
        return combined

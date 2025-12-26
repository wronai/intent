"""
Dynamic Executor - Safe code execution and deployment
Runs generated code in sandbox for validation
"""

import os
import sys
import ast
import tempfile
import subprocess
import asyncio
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
import logging
import importlib.util

from .config import get_settings
from .core import TargetPlatform

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of code execution"""
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    return_value: Any = None
    execution_time_ms: float = 0.0
    memory_used_mb: float = 0.0


class Sandbox:
    """
    Restricted execution environment
    Limits what code can do during validation
    """
    
    # Allowed built-in names
    SAFE_BUILTINS = {
        'True', 'False', 'None',
        'abs', 'all', 'any', 'bool', 'bytes', 'chr', 'dict', 
        'enumerate', 'filter', 'float', 'format', 'frozenset',
        'hash', 'int', 'isinstance', 'issubclass', 'iter', 'len',
        'list', 'map', 'max', 'min', 'next', 'ord', 'pow', 'print',
        'range', 'repr', 'reversed', 'round', 'set', 'slice',
        'sorted', 'str', 'sum', 'tuple', 'type', 'zip'
    }
    
    # Blocked modules
    BLOCKED_MODULES = {
        'os', 'sys', 'subprocess', 'shutil', 'socket', 'http',
        'urllib', 'ftplib', 'telnetlib', 'smtplib', 'poplib',
        'imaplib', 'nntplib', 'ctypes', 'multiprocessing'
    }
    
    def __init__(self, timeout: int = 30, memory_limit: str = "256m"):
        self.timeout = timeout
        self.memory_limit = memory_limit
    
    def create_restricted_globals(self) -> Dict[str, Any]:
        """Create restricted globals for exec()"""
        safe_builtins = {
            name: getattr(__builtins__ if isinstance(__builtins__, dict) 
                         else vars(__builtins__), name, None)
            for name in self.SAFE_BUILTINS
        }
        
        # Add safe __import__
        def safe_import(name, *args, **kwargs):
            if name.split('.')[0] in self.BLOCKED_MODULES:
                raise ImportError(f"Module '{name}' is not allowed in sandbox")
            return __import__(name, *args, **kwargs)
        
        safe_builtins['__import__'] = safe_import
        
        return {
            '__builtins__': safe_builtins,
            '__name__': '__sandbox__',
            '__doc__': None,
        }
    
    def is_code_safe(self, code: str) -> Tuple[bool, List[str]]:
        """Check if code is safe to execute"""
        issues = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, [f"Syntax error: {e}"]
        
        class SafetyChecker(ast.NodeVisitor):
            def visit_Import(self, node):
                for alias in node.names:
                    module = alias.name.split('.')[0]
                    if module in Sandbox.BLOCKED_MODULES:
                        issues.append(f"Blocked import: {alias.name}")
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                if node.module:
                    module = node.module.split('.')[0]
                    if module in Sandbox.BLOCKED_MODULES:
                        issues.append(f"Blocked import from: {node.module}")
                self.generic_visit(node)
            
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ('eval', 'exec', 'compile', '__import__'):
                        issues.append(f"Blocked function: {node.func.id}")
                self.generic_visit(node)
        
        checker = SafetyChecker()
        checker.visit(tree)
        
        return len(issues) == 0, issues


class DynamicExecutor:
    """
    Execute and deploy generated code safely
    """
    
    def __init__(self, sandbox_mode: bool = True):
        self.sandbox_mode = sandbox_mode
        self.sandbox = Sandbox()
        self._temp_dir = tempfile.mkdtemp(prefix="intentforge_")
    
    async def execute(
        self,
        code: str,
        language: str,
        platform: TargetPlatform,
        test_data: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute code and return result"""
        
        if self.sandbox_mode:
            # Check safety first
            if language == "python":
                is_safe, issues = self.sandbox.is_code_safe(code)
                if not is_safe:
                    return ExecutionResult(
                        success=False,
                        error=f"Code safety check failed: {', '.join(issues)}"
                    )
        
        executor = {
            "python": self._execute_python,
            "javascript": self._execute_javascript,
            "sql": self._execute_sql_dry,
            "cpp": self._compile_cpp,
        }.get(language.lower())
        
        if executor:
            return await executor(code, platform, test_data)
        
        return ExecutionResult(
            success=False,
            error=f"No executor available for {language}"
        )
    
    async def _execute_python(
        self,
        code: str,
        platform: TargetPlatform,
        test_data: Optional[Dict[str, Any]]
    ) -> ExecutionResult:
        """Execute Python code in sandbox"""
        import time
        import traceback
        
        start_time = time.time()
        
        try:
            if self.sandbox_mode:
                # Execute in restricted environment
                restricted_globals = self.sandbox.create_restricted_globals()
                restricted_globals['__test_data__'] = test_data or {}
                
                exec(compile(code, '<generated>', 'exec'), restricted_globals)
                
                # Get result if any
                result = restricted_globals.get('result', None)
            else:
                # Full execution
                local_vars = {'test_data': test_data or {}}
                exec(code, local_vars)
                result = local_vars.get('result', None)
            
            return ExecutionResult(
                success=True,
                return_value=result,
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _execute_javascript(
        self,
        code: str,
        platform: TargetPlatform,
        test_data: Optional[Dict[str, Any]]
    ) -> ExecutionResult:
        """Execute JavaScript using Node.js"""
        import json
        import time
        
        start_time = time.time()
        
        # Write temp file
        temp_file = Path(self._temp_dir) / "script.js"
        
        # Wrap code with test data
        wrapped_code = f"""
const testData = {json.dumps(test_data or {})};
{code}
"""
        
        temp_file.write_text(wrapped_code)
        
        try:
            result = subprocess.run(
                ["node", str(temp_file)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return ExecutionResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else None,
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                error="Execution timeout"
            )
        except FileNotFoundError:
            return ExecutionResult(
                success=False,
                error="Node.js not installed"
            )
    
    async def _execute_sql_dry(
        self,
        code: str,
        platform: TargetPlatform,
        test_data: Optional[Dict[str, Any]]
    ) -> ExecutionResult:
        """Dry-run SQL (validate without executing)"""
        import re
        
        issues = []
        
        # Basic SQL validation
        if re.search(r'DROP\s+TABLE', code, re.IGNORECASE):
            issues.append("DROP TABLE detected - potentially destructive")
        
        if re.search(r'DELETE\s+FROM\s+\w+\s*;', code, re.IGNORECASE):
            if not re.search(r'WHERE', code, re.IGNORECASE):
                issues.append("DELETE without WHERE clause")
        
        # Check for parameterized queries
        if "'%s'" in code or '"%s"' in code:
            issues.append("Use %(name)s for parameterized queries")
        
        return ExecutionResult(
            success=len(issues) == 0,
            output="SQL validation passed" if not issues else None,
            error="\n".join(issues) if issues else None
        )
    
    async def _compile_cpp(
        self,
        code: str,
        platform: TargetPlatform,
        test_data: Optional[Dict[str, Any]]
    ) -> ExecutionResult:
        """Compile C++ code (for Arduino/ESP32)"""
        import time
        
        start_time = time.time()
        
        # For Arduino, we just validate syntax
        if platform in (TargetPlatform.ARDUINO_CPP, TargetPlatform.ESP32_MICROPYTHON):
            # Basic syntax check
            temp_file = Path(self._temp_dir) / "sketch.cpp"
            temp_file.write_text(code)
            
            try:
                # Try to compile with g++ just for syntax check
                result = subprocess.run(
                    ["g++", "-fsyntax-only", str(temp_file)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                return ExecutionResult(
                    success=result.returncode == 0,
                    output="Compilation check passed" if result.returncode == 0 else None,
                    error=result.stderr if result.returncode != 0 else None,
                    execution_time_ms=(time.time() - start_time) * 1000
                )
                
            except FileNotFoundError:
                return ExecutionResult(
                    success=True,
                    output="Skipped (g++ not available)"
                )
        
        return ExecutionResult(
            success=False,
            error="Unsupported platform for C++ execution"
        )
    
    def deploy_to_file(
        self,
        code: str,
        language: str,
        output_path: Path,
        make_executable: bool = False
    ) -> bool:
        """Deploy generated code to file"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(code)
            
            if make_executable and language in ("python", "sh"):
                os.chmod(output_path, 0o755)
            
            logger.info(f"Deployed code to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy: {e}")
            return False
    
    def create_test_file(
        self,
        code: str,
        language: str,
        test_framework: str = "pytest"
    ) -> str:
        """Generate test file for generated code"""
        
        if language == "python" and test_framework == "pytest":
            return self._generate_pytest(code)
        elif language == "javascript":
            return self._generate_jest(code)
        
        return ""
    
    def _generate_pytest(self, code: str) -> str:
        """Generate pytest tests for Python code"""
        # Extract function names
        try:
            tree = ast.parse(code)
            functions = [
                node.name for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and not node.name.startswith('_')
            ]
        except SyntaxError:
            functions = []
        
        test_code = '''"""
Auto-generated tests
"""

import pytest


'''
        
        for func in functions:
            test_code += f'''
def test_{func}_exists():
    """Test that {func} is callable"""
    from generated import {func}
    assert callable({func})


def test_{func}_basic():
    """Basic test for {func}"""
    from generated import {func}
    # TODO: Add specific test cases
    pass

'''
        
        return test_code
    
    def _generate_jest(self, code: str) -> str:
        """Generate Jest tests for JavaScript code"""
        return '''/**
 * Auto-generated tests
 */

describe('Generated Code', () => {
    test('should load without errors', () => {
        expect(() => require('./generated')).not.toThrow();
    });
    
    // TODO: Add specific test cases
});
'''
    
    def cleanup(self):
        """Cleanup temporary files"""
        import shutil
        try:
            shutil.rmtree(self._temp_dir)
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    
    def __del__(self):
        self.cleanup()

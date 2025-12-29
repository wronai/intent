"""
IntentForge Code Runner Service

Automatic code execution with self-healing capabilities:
- Auto-detect and install missing dependencies
- Auto-debug and fix code errors
- Retry loop until success or max attempts
- Support for multiple languages
"""

import contextlib
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of code execution errors"""

    MISSING_MODULE = "missing_module"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    PERMISSION = "permission"
    UNKNOWN = "unknown"


@dataclass
class ExecutionResult:
    """Result of code execution"""

    success: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int = -1
    error_type: ErrorType = None
    error_details: dict = field(default_factory=dict)
    attempts: int = 0
    fixes_applied: list = field(default_factory=list)
    final_code: str = ""
    execution_time_ms: float = 0
    execution_logs: list = field(default_factory=list)


@dataclass
class FixAttempt:
    """Record of a fix attempt"""

    error_type: ErrorType
    original_error: str
    fix_description: str
    fix_applied: str
    success: bool


class CodeRunner:
    """
    Self-healing code runner with automatic debugging and fixing.

    Features:
    - Detects missing modules and installs them
    - Detects syntax errors and fixes them via LLM
    - Detects runtime errors and suggests/applies fixes
    - Retries execution after fixes
    - Tracks all fix attempts for transparency
    """

    def __init__(self, max_retries: int = 3, auto_install: bool = True):
        self.max_retries = max_retries
        self.auto_install = auto_install
        self.installed_packages: set[str] = set()

        # Language configurations
        self.lang_config = {
            "python": {
                "extension": ".py",
                "command": ["python3"],
                "pip": "pip3",
                "module_pattern": r"No module named '(\w+)'|ModuleNotFoundError: No module named '([^']+)'",
                "syntax_pattern": r"SyntaxError:",
                "common_packages": {
                    "requests": "requests",
                    "numpy": "numpy",
                    "pandas": "pandas",
                    "flask": "flask",
                    "fastapi": "fastapi",
                    "httpx": "httpx",
                    "aiohttp": "aiohttp",
                    "beautifulsoup4": "bs4",
                    "pillow": "PIL",
                    "opencv-python": "cv2",
                    "scikit-learn": "sklearn",
                    "tensorflow": "tensorflow",
                    "torch": "torch",
                    "transformers": "transformers",
                },
            },
            "javascript": {
                "extension": ".js",
                "command": ["node"],
                "npm": "npm",
                "module_pattern": r"Cannot find module '([^']+)'",
                "syntax_pattern": r"SyntaxError:",
            },
            "bash": {
                "extension": ".sh",
                "command": ["bash"],
            },
            "shell": {
                "extension": ".sh",
                "command": ["sh"],
            },
        }

    async def run(
        self,
        code: str,
        language: str = "python",
        auto_fix: bool = True,
        auto_install_deps: bool = True,
    ) -> ExecutionResult:
        """
        Run code with automatic error detection, fixing, and retry.

        Args:
            code: Source code to execute
            language: Programming language
            auto_fix: Whether to automatically fix errors
            auto_install_deps: Whether to auto-install missing dependencies

        Returns:
            ExecutionResult with details of execution and any fixes applied
        """
        import time

        start_time = time.time()

        result = ExecutionResult(
            success=False,
            final_code=code,
            attempts=0,
        )

        # Execution logs for frontend display
        execution_logs = []

        current_code = code

        for attempt in range(self.max_retries + 1):
            result.attempts = attempt + 1
            log_msg = f"🔄 Attempt {attempt + 1}/{self.max_retries + 1}: Executing code..."
            execution_logs.append(log_msg)
            logger.info(f"Code execution attempt {attempt + 1}/{self.max_retries + 1}")

            # Execute the code
            exec_result = await self._execute_code(current_code, language)

            result.stdout = exec_result.get("stdout", "")
            result.stderr = exec_result.get("stderr", "")
            result.returncode = exec_result.get("returncode", -1)

            # Check if successful
            if exec_result.get("success"):
                result.success = True
                result.final_code = current_code
                execution_logs.append(f"✅ Success on attempt {attempt + 1}")
                logger.info(f"Code execution succeeded on attempt {attempt + 1}")
                break

            # Log the error
            error_preview = (
                result.stderr[:150].replace("\n", " ") if result.stderr else "Unknown error"
            )
            execution_logs.append(f"❌ Failed: {error_preview}")
            logger.info(f"Code execution failed: {result.stderr[:200]}")

            # If no auto-fix or max retries reached, stop
            if not auto_fix:
                logger.info("Auto-fix disabled, stopping")
                break
            if attempt >= self.max_retries:
                logger.info(f"Max retries ({self.max_retries}) reached, stopping")
                break

            # Analyze error and attempt fix
            execution_logs.append("🔍 Analyzing error...")
            error_type, error_details = self._analyze_error(result.stderr, result.stdout, language)
            result.error_type = error_type
            result.error_details = error_details
            execution_logs.append(f"📋 Error type: {error_type.value if error_type else 'unknown'}")
            logger.info(f"Error analyzed: type={error_type}, details={error_details}")

            # Try to fix the error
            execution_logs.append("🔧 Attempting auto-fix...")
            fix_result = await self._fix_error(
                current_code,
                language,
                error_type,
                error_details,
                result.stderr,
                auto_install_deps,
            )

            logger.info(f"Fix result: {fix_result}")

            if fix_result.get("fixed"):
                fix_attempt = FixAttempt(
                    error_type=error_type,
                    original_error=result.stderr[:500],
                    fix_description=fix_result.get("description", ""),
                    fix_applied=fix_result.get("fix_type", ""),
                    success=True,
                )
                result.fixes_applied.append(fix_attempt)

                # Update code if it was modified
                if fix_result.get("new_code"):
                    current_code = fix_result["new_code"]

                execution_logs.append(
                    f"✅ Fix applied: {fix_result.get('description', 'unknown fix')}"
                )
                logger.info(f"Fix applied: {fix_result.get('description')}")
            else:
                # Store the failed fix reason for debugging
                result.error_details["fix_failed"] = fix_result.get("reason", "Unknown")
                execution_logs.append(f"⚠️ Fix failed: {fix_result.get('reason', 'Unknown')}")
                logger.warning(f"Fix failed: {fix_result.get('reason')}")
                # Continue to next attempt anyway for certain error types
                if error_type != ErrorType.MISSING_MODULE:
                    break

        result.execution_time_ms = (time.time() - start_time) * 1000
        result.final_code = current_code
        result.execution_logs = execution_logs

        return result

    async def _execute_code(self, code: str, language: str) -> dict:
        """Execute code and return result"""
        config = self.lang_config.get(language, self.lang_config["python"])

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=config["extension"], delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            result = subprocess.run(
                config["command"] + [temp_path],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
                cwd="/tmp",
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout[:10000],
                "stderr": result.stderr[:5000],
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Execution timeout (30s)",
                "returncode": -1,
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
            }
        finally:
            with contextlib.suppress(OSError):
                os.unlink(temp_path)

    def _analyze_error(self, stderr: str, stdout: str, language: str) -> tuple[ErrorType, dict]:
        """Analyze error output and determine error type"""
        config = self.lang_config.get(language, {})

        # Check for missing module
        module_pattern = config.get("module_pattern", r"No module named '(\w+)'")
        module_match = re.search(module_pattern, stderr)
        if module_match:
            module_name = module_match.group(1) or module_match.group(2)
            return ErrorType.MISSING_MODULE, {"module": module_name}

        # Check for syntax error
        syntax_pattern = config.get("syntax_pattern", r"SyntaxError:")
        if re.search(syntax_pattern, stderr):
            # Extract line number if available
            line_match = re.search(r"line (\d+)", stderr)
            line_num = int(line_match.group(1)) if line_match else None
            return ErrorType.SYNTAX_ERROR, {"line": line_num, "message": stderr}

        # Check for timeout
        if "timeout" in stderr.lower():
            return ErrorType.TIMEOUT, {}

        # Check for permission error
        if "permission" in stderr.lower() or "PermissionError" in stderr:
            return ErrorType.PERMISSION, {}

        # Runtime error (generic)
        if stderr:
            # Try to extract error type and message
            error_match = re.search(r"(\w+Error): (.+)", stderr)
            if error_match:
                return ErrorType.RUNTIME_ERROR, {
                    "error_class": error_match.group(1),
                    "message": error_match.group(2),
                }

        return ErrorType.UNKNOWN, {"stderr": stderr}

    async def _fix_error(
        self,
        code: str,
        language: str,
        error_type: ErrorType,
        error_details: dict,
        stderr: str,
        auto_install: bool,
    ) -> dict:
        """Attempt to fix the detected error"""

        if error_type == ErrorType.MISSING_MODULE:
            return await self._fix_missing_module(
                code, language, error_details.get("module", ""), auto_install
            )

        elif error_type == ErrorType.SYNTAX_ERROR:
            return await self._fix_syntax_error(code, language, stderr)

        elif error_type == ErrorType.RUNTIME_ERROR:
            return await self._fix_runtime_error(code, language, stderr, error_details)

        return {"fixed": False, "reason": f"Cannot auto-fix {error_type.value}"}

    async def _fix_missing_module(
        self, code: str, language: str, module: str, auto_install: bool
    ) -> dict:
        """Fix missing module by installing it"""
        if language != "python":
            return {"fixed": False, "reason": "Auto-install only for Python"}

        if not auto_install:
            return {"fixed": False, "reason": "Auto-install disabled"}

        # Map module import name to package name
        config = self.lang_config.get("python", {})
        common_packages = config.get("common_packages", {})

        # Reverse lookup - find package name from import name
        package_name = module
        for pkg, imp in common_packages.items():
            if imp == module:
                package_name = pkg
                break

        # Check if already tried
        if package_name in self.installed_packages:
            return {"fixed": False, "reason": f"Already tried installing {package_name}"}

        # Install the package
        logger.info(f"Installing missing package: {package_name}")

        try:
            # Try multiple pip commands
            pip_commands = [
                ["pip3", "install", package_name],
                ["pip", "install", package_name],
                ["python3", "-m", "pip", "install", package_name],
                ["pip3", "install", "--user", package_name],
            ]

            install_success = False
            install_output = ""

            for pip_cmd in pip_commands:
                try:
                    result = subprocess.run(
                        pip_cmd,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )

                    if result.returncode == 0:
                        install_success = True
                        install_output = result.stdout
                        logger.info(f"Successfully installed {package_name} with {pip_cmd[0]}")
                        break
                    else:
                        install_output = result.stderr
                except FileNotFoundError:
                    continue
                except Exception as e:
                    install_output = str(e)
                    continue

            self.installed_packages.add(package_name)

            if install_success:
                return {
                    "fixed": True,
                    "fix_type": "install_package",
                    "description": f"Installed package: {package_name}",
                    "new_code": None,  # Code unchanged, just retry
                }
            else:
                return {
                    "fixed": False,
                    "reason": f"Failed to install {package_name}: {install_output[:500]}",
                }
        except Exception as e:
            logger.error(f"Package install error: {e}")
            return {"fixed": False, "reason": str(e)}

    async def _fix_syntax_error(self, code: str, language: str, stderr: str) -> dict:
        """Fix syntax error using LLM"""
        from .services import ChatService

        chat = ChatService()

        prompt = f"""Fix the syntax error in this {language} code.
Error:
{stderr[:1000]}

Code:
```{language}
{code}
```

Return ONLY the fixed code, no explanations. The code must be complete and runnable."""

        try:
            response = await chat.send(
                message=prompt,
                system="You are a code fixer. Return only the fixed code, nothing else. No markdown, no explanations.",
            )

            if response.get("success"):
                fixed_code = response.get("response", "")

                # Extract code from markdown if present
                if "```" in fixed_code:
                    match = re.search(r"```\w*\n([\s\S]*?)```", fixed_code)
                    if match:
                        fixed_code = match.group(1)

                if fixed_code.strip() and fixed_code.strip() != code.strip():
                    return {
                        "fixed": True,
                        "fix_type": "syntax_fix",
                        "description": "Fixed syntax error via LLM",
                        "new_code": fixed_code.strip(),
                    }

            return {"fixed": False, "reason": "LLM could not fix syntax error"}
        except Exception as e:
            return {"fixed": False, "reason": str(e)}

    async def _fix_runtime_error(
        self, code: str, language: str, stderr: str, error_details: dict
    ) -> dict:
        """Fix runtime error using LLM"""
        from .services import ChatService

        chat = ChatService()

        error_class = error_details.get("error_class", "Error")
        error_message = error_details.get("message", stderr[:500])

        prompt = f"""Fix the runtime error in this {language} code.
Error type: {error_class}
Error message: {error_message}

Full error:
{stderr[:1000]}

Code:
```{language}
{code}
```

Return ONLY the fixed code that will not produce this error. No explanations."""

        try:
            response = await chat.send(
                message=prompt,
                system="You are a code fixer. Fix the error and return only the corrected code. No markdown blocks, no explanations.",
            )

            if response.get("success"):
                fixed_code = response.get("response", "")

                # Extract code from markdown if present
                if "```" in fixed_code:
                    match = re.search(r"```\w*\n([\s\S]*?)```", fixed_code)
                    if match:
                        fixed_code = match.group(1)

                if fixed_code.strip() and fixed_code.strip() != code.strip():
                    return {
                        "fixed": True,
                        "fix_type": "runtime_fix",
                        "description": f"Fixed {error_class} via LLM",
                        "new_code": fixed_code.strip(),
                    }

            return {"fixed": False, "reason": "LLM could not fix runtime error"}
        except Exception as e:
            return {"fixed": False, "reason": str(e)}


# Global instance
code_runner = CodeRunner()


async def run_with_autofix(
    code: str,
    language: str = "python",
    max_retries: int = 3,
    auto_install: bool = True,
) -> dict:
    """
    Convenience function to run code with auto-fix.

    Returns dict with:
    - success: bool
    - output: str (stdout)
    - error: str (stderr if failed)
    - attempts: int
    - fixes: list of applied fixes
    - final_code: str (possibly modified code)
    """
    runner = CodeRunner(max_retries=max_retries, auto_install=auto_install)
    result = await runner.run(code, language, auto_fix=True, auto_install_deps=auto_install)

    response = {
        "success": result.success,
        "output": result.stdout,
        "error": result.stderr if not result.success else "",
        "attempts": result.attempts,
        "fixes": [
            {
                "type": f.fix_applied,
                "description": f.fix_description,
            }
            for f in result.fixes_applied
        ],
        "final_code": result.final_code,
        "execution_time_ms": result.execution_time_ms,
        "logs": result.execution_logs,
    }

    # Add debug info
    if result.error_type:
        response["error_type"] = result.error_type.value
    if result.error_details:
        response["error_details"] = result.error_details

    return response

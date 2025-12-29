"""
IntentForge Code Tester Service

Automatic test generation and execution with self-healing:
- Generate tests based on code intent/requirements
- Run tests and detect failures
- Fix code iteratively until all tests pass
- Support for multiple test frameworks
- Sandbox environment management
"""

import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class TestFramework(Enum):
    """Supported test frameworks"""

    PYTEST = "pytest"
    UNITTEST = "unittest"
    DOCTEST = "doctest"
    SIMPLE = "simple"  # Basic assert-based testing


class TestStatus(Enum):
    """Test execution status"""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class TestCase:
    """Single test case"""

    name: str
    code: str
    expected_output: str = ""
    status: TestStatus = TestStatus.SKIPPED
    actual_output: str = ""
    error_message: str = ""


@dataclass
class TestResult:
    """Result of test execution"""

    success: bool
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    test_cases: list = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    execution_time_ms: float = 0
    logs: list = field(default_factory=list)


@dataclass
class CodeWithTests:
    """Code bundle with tests"""

    source_code: str
    test_code: str
    requirements: list = field(default_factory=list)
    intent: str = ""
    language: str = "python"


class SandboxEnvironment:
    """
    Manages isolated sandbox environments for code testing.
    Creates temporary virtual environments or Docker containers.
    """

    def __init__(self, use_docker: bool = False):
        self.use_docker = use_docker
        self.env_path: Path = None
        self.active = False

    async def create(self, requirements: list | None = None) -> bool:
        """Create sandbox environment"""
        try:
            if self.use_docker:
                return await self._create_docker_sandbox(requirements)
            else:
                return await self._create_venv_sandbox(requirements)
        except Exception as e:
            logger.error(f"Failed to create sandbox: {e}")
            return False

    async def _create_venv_sandbox(self, requirements: list | None = None) -> bool:
        """Create virtual environment sandbox"""
        self.env_path = Path(tempfile.mkdtemp(prefix="intentforge_sandbox_"))
        venv_path = self.env_path / "venv"

        # Create venv
        result = subprocess.run(
            ["python3", "-m", "venv", str(venv_path)],
            check=False,
            capture_output=True,
            timeout=60,
        )

        if result.returncode != 0:
            logger.error(f"Failed to create venv: {result.stderr.decode()}")
            return False

        # Install requirements
        if requirements:
            pip_path = venv_path / "bin" / "pip"
            for req in requirements:
                subprocess.run(
                    [str(pip_path), "install", req],
                    check=False,
                    capture_output=True,
                    timeout=120,
                )

        # Install pytest for testing
        pip_path = venv_path / "bin" / "pip"
        subprocess.run(
            [str(pip_path), "install", "pytest"],
            check=False,
            capture_output=True,
            timeout=60,
        )

        self.active = True
        return True

    async def _create_docker_sandbox(self, requirements: list | None = None) -> bool:
        """Create Docker container sandbox"""
        # Use existing IntentForge Docker setup
        self.active = True
        return True

    def get_python_path(self) -> str:
        """Get path to Python interpreter in sandbox"""
        if self.use_docker:
            return "python3"
        if self.env_path:
            return str(self.env_path / "venv" / "bin" / "python")
        return "python3"

    async def cleanup(self):
        """Clean up sandbox environment"""
        if self.env_path and self.env_path.exists():
            import shutil

            shutil.rmtree(self.env_path, ignore_errors=True)
        self.active = False


class CodeTester:
    """
    Test-driven code generation and fixing.

    Flow:
    1. Generate tests from intent/requirements
    2. Run tests against code
    3. If tests fail, analyze failures and fix code
    4. Repeat until all tests pass or max iterations
    """

    def __init__(self, max_iterations: int = 5, use_sandbox: bool = True):
        self.max_iterations = max_iterations
        self.use_sandbox = use_sandbox
        self.sandbox: SandboxEnvironment = None

    async def test_and_fix(
        self,
        code: str,
        intent: str,
        tests: str | None = None,
        language: str = "python",
        requirements: list | None = None,
    ) -> dict:
        """
        Test code and fix until all tests pass.

        Args:
            code: Source code to test
            intent: What the code should do (for test generation)
            tests: Optional pre-written tests
            language: Programming language
            requirements: Package requirements

        Returns:
            dict with success status, final code, test results, and logs
        """
        import time

        start_time = time.time()
        logs = []

        logs.append("🧪 Starting test-driven development loop")
        logs.append(f"📋 Intent: {intent[:100]}...")

        # Generate tests if not provided
        if not tests:
            logs.append("📝 Generating tests from intent...")
            tests = await self._generate_tests(code, intent, language)
            logs.append(f"✅ Generated {tests.count('def test_')} test functions")

        # Setup sandbox if needed
        if self.use_sandbox:
            logs.append("🔒 Creating sandbox environment...")
            self.sandbox = SandboxEnvironment(use_docker=False)
            if await self.sandbox.create(requirements):
                logs.append("✅ Sandbox ready")
            else:
                logs.append("⚠️ Sandbox creation failed, using system Python")
                self.sandbox = None

        current_code = code
        iteration = 0
        all_passed = False
        test_results = []

        try:
            while iteration < self.max_iterations and not all_passed:
                iteration += 1
                logs.append(f"\n🔄 Iteration {iteration}/{self.max_iterations}")

                # Run tests
                logs.append("▶️ Running tests...")
                result = await self._run_tests(current_code, tests, language)
                test_results.append(result)

                logs.append(f"📊 Results: {result.passed}/{result.total_tests} passed")

                if result.success:
                    all_passed = True
                    logs.append("✅ All tests passed!")
                    break

                # Analyze failures and fix
                if result.failed > 0 or result.errors > 0:
                    logs.append(f"❌ {result.failed} failed, {result.errors} errors")
                    logs.append("🔧 Attempting to fix code...")

                    fix_result = await self._fix_code_for_tests(
                        current_code,
                        tests,
                        result,
                        intent,
                        language,
                    )

                    if fix_result.get("fixed"):
                        current_code = fix_result["new_code"]
                        logs.append(
                            f"✅ Code fixed: {fix_result.get('description', 'applied fix')}"
                        )
                    else:
                        logs.append(f"⚠️ Could not fix: {fix_result.get('reason', 'unknown')}")
                        break

        finally:
            # Cleanup sandbox
            if self.sandbox:
                await self.sandbox.cleanup()

        execution_time_ms = (time.time() - start_time) * 1000

        return {
            "success": all_passed,
            "final_code": current_code,
            "tests": tests,
            "iterations": iteration,
            "test_results": {
                "total": test_results[-1].total_tests if test_results else 0,
                "passed": test_results[-1].passed if test_results else 0,
                "failed": test_results[-1].failed if test_results else 0,
            },
            "logs": logs,
            "execution_time_ms": execution_time_ms,
        }

    async def _generate_tests(self, code: str, intent: str, language: str) -> str:
        """Generate tests based on code and intent using LLM"""
        from .services import ChatService

        chat = ChatService()

        prompt = f"""Generate pytest test cases for the following Python code based on the intent.

INTENT: {intent}

CODE:
```python
{code}
```

Requirements:
1. Generate at least 3 test functions
2. Test normal cases, edge cases, and error handling
3. Use pytest style (def test_xxx():)
4. Include assertions with clear expected values
5. Tests should be self-contained

Return ONLY the test code, no explanations.
Example format:
```python
import pytest

def test_normal_case():
    result = function_to_test(input)
    assert result == expected

def test_edge_case():
    ...
```
"""

        response = await chat.send(
            message=prompt,
            system="You are a test engineer. Generate comprehensive pytest test cases. Return only Python code.",
        )

        if not response.get("success"):
            # Fallback to basic test
            return self._generate_basic_tests(code, intent)

        test_code = response.get("response", "")

        # Extract code from markdown
        code_match = re.search(r"```python\n([\s\S]*?)```", test_code)
        if code_match:
            return code_match.group(1)

        return test_code

    def _generate_basic_tests(self, code: str, intent: str) -> str:
        """Generate basic tests when LLM is unavailable"""
        # Extract function names from code
        func_matches = re.findall(r"def\s+(\w+)\s*\(", code)

        tests = ["import pytest", ""]

        for func in func_matches:
            if not func.startswith("_"):
                tests.append(f"""
def test_{func}_exists():
    \"\"\"Test that {func} function exists and is callable\"\"\"
    assert callable({func})

def test_{func}_basic():
    \"\"\"Basic test for {func}\"\"\"
    # Add actual test based on function signature
    pass
""")

        return "\n".join(tests)

    async def _run_tests(
        self,
        code: str,
        tests: str,
        language: str,
    ) -> TestResult:
        """Run tests against code"""
        import time

        start_time = time.time()

        result = TestResult(success=False)

        # Create temporary directory with code and tests
        with tempfile.TemporaryDirectory(prefix="intentforge_test_") as tmpdir:
            # Write source code
            code_file = Path(tmpdir) / "source_code.py"
            code_file.write_text(code)

            # Write tests with import
            test_file = Path(tmpdir) / "test_code.py"
            test_content = f"""
import sys
sys.path.insert(0, '{tmpdir}')
from source_code import *

{tests}
"""
            test_file.write_text(test_content)

            # Get Python path
            python_path = self.sandbox.get_python_path() if self.sandbox else "python3"

            # Run pytest
            try:
                proc = subprocess.run(
                    [python_path, "-m", "pytest", str(test_file), "-v", "--tb=short"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=tmpdir,
                )

                result.stdout = proc.stdout
                result.stderr = proc.stderr

                # Parse pytest output
                result = self._parse_pytest_output(proc.stdout, proc.stderr, result)

            except subprocess.TimeoutExpired:
                result.stderr = "Test execution timed out"
                result.errors = 1
            except Exception as e:
                result.stderr = str(e)
                result.errors = 1

        result.execution_time_ms = (time.time() - start_time) * 1000
        return result

    def _parse_pytest_output(self, stdout: str, stderr: str, result: TestResult) -> TestResult:
        """Parse pytest output to extract test results"""
        # Look for summary line: "X passed, Y failed, Z errors"
        # Count passed/failed from output
        passed_match = re.search(r"(\d+)\s+passed", stdout)
        failed_match = re.search(r"(\d+)\s+failed", stdout)
        error_match = re.search(r"(\d+)\s+error", stdout)

        result.passed = int(passed_match.group(1)) if passed_match else 0
        result.failed = int(failed_match.group(1)) if failed_match else 0
        result.errors = int(error_match.group(1)) if error_match else 0

        # Check for collection errors
        if "error" in stderr.lower() or "ModuleNotFoundError" in stdout:
            result.errors += 1

        result.total_tests = result.passed + result.failed + result.errors
        result.success = result.failed == 0 and result.errors == 0 and result.passed > 0

        return result

    async def _fix_code_for_tests(
        self,
        code: str,
        tests: str,
        test_result: TestResult,
        intent: str,
        language: str,
    ) -> dict:
        """Fix code based on test failures"""
        from .services import ChatService

        chat = ChatService()

        prompt = f"""Fix the Python code to pass all tests.

INTENT: {intent}

CURRENT CODE:
```python
{code}
```

TESTS:
```python
{tests}
```

TEST OUTPUT (failures):
```
{test_result.stdout[-2000:] if test_result.stdout else ""}
{test_result.stderr[-1000:] if test_result.stderr else ""}
```

Fix the code so all tests pass. Return ONLY the fixed Python code, no explanations.
"""

        response = await chat.send(
            message=prompt,
            system="You are a Python expert. Fix code to pass tests. Return only the corrected Python code.",
        )

        if not response.get("success"):
            return {"fixed": False, "reason": "LLM request failed"}

        fixed_code = response.get("response", "")

        # Extract code from markdown
        code_match = re.search(r"```python\n([\s\S]*?)```", fixed_code)
        if code_match:
            fixed_code = code_match.group(1)

        if fixed_code and fixed_code != code:
            return {
                "fixed": True,
                "new_code": fixed_code,
                "description": "Code fixed based on test failures",
            }

        return {"fixed": False, "reason": "No changes made to code"}


# Global instance
code_tester = CodeTester()


async def test_and_fix_code(
    code: str,
    intent: str,
    tests: str | None = None,
    language: str = "python",
    max_iterations: int = 5,
) -> dict:
    """
    Convenience function to test and fix code.

    Args:
        code: Source code to test
        intent: What the code should do
        tests: Optional pre-written tests
        language: Programming language
        max_iterations: Max fix attempts

    Returns:
        dict with success, final_code, test_results, logs
    """
    tester = CodeTester(max_iterations=max_iterations)
    return await tester.test_and_fix(
        code=code,
        intent=intent,
        tests=tests,
        language=language,
    )

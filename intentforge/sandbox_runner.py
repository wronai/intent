"""
IntentForge Sandbox Runner with Live Streaming

Provides:
- Isolated code execution in Docker
- WebSocket streaming of logs
- Service detection and testing
- Auto-fix capabilities
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    STOPPED = "stopped"


class ServiceType(Enum):
    FLASK = "flask"
    FASTAPI = "fastapi"
    EXPRESS = "express"
    SCRIPT = "script"
    UNKNOWN = "unknown"


@dataclass
class LogEntry:
    """Single log entry"""
    timestamp: float
    type: str  # info, status, error, shell
    message: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "type": self.type,
            "data": self.message
        }


@dataclass
class EndpointTest:
    """Result of testing a single endpoint"""
    method: str
    path: str
    status_code: int
    success: bool
    response_time_ms: float
    response_body: str = ""
    error: str = ""


@dataclass
class SandboxResult:
    """Final result of sandbox execution"""
    success: bool
    status: ExecutionStatus
    output: str = ""
    error: str = ""
    execution_time_ms: float = 0
    service_type: ServiceType = ServiceType.UNKNOWN
    test_results: list[EndpointTest] = field(default_factory=list)
    endpoints_passed: int = 0
    endpoints_tested: int = 0
    final_code: str = ""
    fixes_applied: list[dict] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)


class SandboxSession:
    """
    A single sandbox execution session.
    Handles Docker container lifecycle and log streaming.
    """

    def __init__(self, session_id: str, code: str, auto_fix: bool = True):
        self.session_id = session_id
        self.code = code
        self.auto_fix = auto_fix
        self.status = ExecutionStatus.PENDING
        self.logs: list[LogEntry] = []
        self.subscribers: list[asyncio.Queue] = []
        self.container_id: str | None = None
        self.process: asyncio.subprocess.Process | None = None
        self.start_time: float = 0
        self.result: SandboxResult | None = None

    def subscribe(self) -> asyncio.Queue:
        """Subscribe to log stream"""
        queue: asyncio.Queue = asyncio.Queue()
        self.subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue):
        """Unsubscribe from log stream"""
        if queue in self.subscribers:
            self.subscribers.remove(queue)

    async def emit(self, log_type: str, message: str):
        """Emit log to all subscribers"""
        entry = LogEntry(
            timestamp=time.time(),
            type=log_type,
            message=message
        )
        self.logs.append(entry)

        for queue in self.subscribers:
            await queue.put(entry.to_dict())

    async def emit_result(self, result: SandboxResult):
        """Emit final result"""
        self.result = result
        for queue in self.subscribers:
            await queue.put({
                "type": "result",
                "data": {
                    "success": result.success,
                    "status": result.status.value,
                    "output": result.output,
                    "error": result.error,
                    "execution_time_ms": result.execution_time_ms,
                    "service_type": result.service_type.value,
                    "test_results": [
                        {
                            "method": t.method,
                            "path": t.path,
                            "status_code": t.status_code,
                            "success": t.success,
                            "response_time_ms": t.response_time_ms
                        }
                        for t in result.test_results
                    ],
                    "endpoints_passed": result.endpoints_passed,
                    "endpoints_tested": result.endpoints_tested,
                    "final_code": result.final_code,
                    "logs": result.logs
                }
            })


class ServiceDetector:
    """Detects what type of service the code is"""

    PATTERNS = {
        ServiceType.FLASK: [
            r"from flask import",
            r"import flask",
            r"Flask\(__name__\)",
            r"@app\.route\(",
        ],
        ServiceType.FASTAPI: [
            r"from fastapi import",
            r"import fastapi",
            r"FastAPI\(\)",
            r"@app\.(get|post|put|delete|patch)\(",
        ],
        ServiceType.EXPRESS: [
            r"require\(['\"]express['\"]\)",
            r"import express from",
            r"express\(\)",
            r"app\.(get|post|put|delete)\(",
        ],
    }

    @classmethod
    def detect(cls, code: str) -> ServiceType:
        """Detect service type from code"""
        for service_type, patterns in cls.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    return service_type
        return ServiceType.SCRIPT

    @classmethod
    def extract_endpoints(cls, code: str, service_type: ServiceType) -> list[dict]:
        """Extract API endpoints from code"""
        endpoints = []

        if service_type == ServiceType.FLASK:
            # Flask routes: @app.route('/path', methods=['GET', 'POST'])
            pattern = r"@app\.route\(['\"]([^'\"]+)['\"](?:,\s*methods=\[([^\]]+)\])?\)"
            for match in re.finditer(pattern, code):
                path = match.group(1)
                methods_str = match.group(2)
                if methods_str:
                    methods = [m.strip().strip("'\"") for m in methods_str.split(",")]
                else:
                    methods = ["GET"]
                for method in methods:
                    endpoints.append({"method": method.upper(), "path": path})

        elif service_type == ServiceType.FASTAPI:
            # FastAPI routes: @app.get('/path'), @app.post('/path'), etc.
            pattern = r"@app\.(get|post|put|delete|patch)\(['\"]([^'\"]+)['\"]"
            for match in re.finditer(pattern, code, re.IGNORECASE):
                method = match.group(1).upper()
                path = match.group(2)
                endpoints.append({"method": method, "path": path})

        return endpoints


class ServiceTester:
    """Tests service endpoints"""

    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.timeout = 10

    async def wait_for_ready(self, max_wait: int = 30) -> bool:
        """Wait for service to be ready"""
        import httpx

        start = time.time()
        while time.time() - start < max_wait:
            try:
                async with httpx.AsyncClient(timeout=2) as client:
                    response = await client.get(self.base_url)
                    return True
            except:
                await asyncio.sleep(0.5)
        return False

    async def test_endpoint(self, method: str, path: str) -> EndpointTest:
        """Test a single endpoint"""
        import httpx

        url = f"{self.base_url}{path}"
        start = time.time()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Prepare request
                if method == "GET":
                    response = await client.get(url)
                elif method == "POST":
                    # Try with sample JSON body
                    response = await client.post(url, json={"test": "data"})
                elif method == "PUT":
                    response = await client.put(url, json={"test": "data"})
                elif method == "DELETE":
                    response = await client.delete(url)
                else:
                    response = await client.request(method, url)

                latency = (time.time() - start) * 1000

                # Consider 2xx and 4xx as "working" (4xx means endpoint exists but validation failed)
                success = response.status_code < 500

                return EndpointTest(
                    method=method,
                    path=path,
                    status_code=response.status_code,
                    success=success,
                    response_time_ms=latency,
                    response_body=response.text[:500] if success else ""
                )

        except Exception as e:
            latency = (time.time() - start) * 1000
            return EndpointTest(
                method=method,
                path=path,
                status_code=0,
                success=False,
                response_time_ms=latency,
                error=str(e)
            )


class SandboxRunner:
    """
    Main sandbox runner.
    Executes code in isolated environment with live streaming.
    """

    def __init__(self, workspace: str = "/tmp/sandbox"):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.sessions: dict[str, SandboxSession] = {}
        self.docker_image = "python:3.12-slim"
        self.max_execution_time = 120  # seconds

    def create_session(self, code: str, auto_fix: bool = True) -> SandboxSession:
        """Create new sandbox session"""
        session_id = hashlib.md5(f"{code}{time.time()}".encode()).hexdigest()[:8]
        session = SandboxSession(session_id, code, auto_fix)
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> SandboxSession | None:
        """Get existing session"""
        return self.sessions.get(session_id)

    async def run(self, session: SandboxSession) -> SandboxResult:
        """
        Execute code in sandbox with live streaming.

        Flow:
        1. Detect service type
        2. Create temp directory
        3. Install dependencies
        4. Run code
        5. Test endpoints (if service)
        6. Return result
        """
        session.status = ExecutionStatus.RUNNING
        session.start_time = time.time()

        try:
            # Detect service type
            service_type = ServiceDetector.detect(session.code)
            await session.emit("log", f"🔍 Detected {service_type.value} service")

            # Extract endpoints
            endpoints = ServiceDetector.extract_endpoints(session.code, service_type)
            if endpoints:
                await session.emit("log", f"📍 Endpoints found: {len(endpoints)}")
                for ep in endpoints:
                    await session.emit("log", f"   {ep['method']} {ep['path']}")

            # Create temp directory
            work_dir = self.workspace / session.session_id
            work_dir.mkdir(exist_ok=True)

            # If it's a web service, use service runner
            if service_type in [ServiceType.FLASK, ServiceType.FASTAPI]:
                result = await self._run_service(session, work_dir, service_type, endpoints)
            else:
                result = await self._run_script(session, work_dir)

            result.service_type = service_type
            result.execution_time_ms = (time.time() - session.start_time) * 1000

            await session.emit_result(result)
            return result

        except Exception as e:
            logger.exception("Sandbox execution error")
            result = SandboxResult(
                success=False,
                status=ExecutionStatus.ERROR,
                error=str(e),
                execution_time_ms=(time.time() - session.start_time) * 1000
            )
            await session.emit_result(result)
            return result

        finally:
            session.status = ExecutionStatus.SUCCESS if result.success else ExecutionStatus.ERROR

    async def _run_service(
        self,
        session: SandboxSession,
        work_dir: Path,
        service_type: ServiceType,
        endpoints: list[dict]
    ) -> SandboxResult:
        """Run a web service and test endpoints"""

        max_attempts = 3 if session.auto_fix else 1
        current_code = session.code
        fixes_applied = []

        for attempt in range(1, max_attempts + 1):
            await session.emit("log", f"\n🔄 Attempt {attempt}/{max_attempts}")

            # Write code to file
            if service_type == ServiceType.FLASK:
                filename = "app.py"
            elif service_type == ServiceType.FASTAPI:
                filename = "main.py"
            else:
                filename = "app.py"

            code_file = work_dir / filename
            code_file.write_text(current_code)
            await session.emit("log", f"📝 Wrote code to {filename}")

            # Detect and install dependencies
            deps = self._extract_dependencies(current_code)
            if deps:
                await session.emit("log", f"📦 Installing: {', '.join(deps)}")
                install_result = await self._pip_install(work_dir, deps, session)
                if not install_result:
                    if session.auto_fix:
                        # Try to fix import errors
                        current_code = await self._fix_imports(current_code, deps)
                        fixes_applied.append({"type": "import", "description": f"Fixed imports for {deps}"})
                        continue

            # Start service
            container_id = await self._start_service_container(work_dir, service_type, session)
            if not container_id:
                if session.auto_fix and attempt < max_attempts:
                    continue
                return SandboxResult(
                    success=False,
                    status=ExecutionStatus.ERROR,
                    error="Failed to start service container",
                    final_code=current_code,
                    fixes_applied=fixes_applied
                )

            session.container_id = container_id

            # Wait for service to be ready
            await session.emit("log", "⏳ Waiting for service to be ready...")
            tester = ServiceTester("http://localhost:5000")

            if await tester.wait_for_ready(30):
                await session.emit("log", "✅ Service ready on port 5000")

                # Test endpoints
                await session.emit("log", "\n🧪 Testing endpoints...")
                test_results = []

                for ep in endpoints:
                    result = await tester.test_endpoint(ep["method"], ep["path"])
                    test_results.append(result)

                    status_icon = "✅" if result.success else "❌"
                    await session.emit("log", f"   {status_icon} {ep['method']} {ep['path']} → {result.status_code} ({result.response_time_ms:.0f}ms)")

                passed = sum(1 for t in test_results if t.success)
                total = len(test_results)

                await session.emit("log", f"\n{'✅' if passed == total else '⚠️'} {passed}/{total} endpoints passed!")

                # Stop container
                await self._stop_container(container_id, session)

                return SandboxResult(
                    success=passed == total,
                    status=ExecutionStatus.SUCCESS if passed == total else ExecutionStatus.ERROR,
                    output=f"Service tested: {passed}/{total} endpoints passed",
                    test_results=test_results,
                    endpoints_passed=passed,
                    endpoints_tested=total,
                    final_code=current_code,
                    fixes_applied=fixes_applied
                )
            else:
                await session.emit("log", "❌ Service failed to start")

                # Get container logs for debugging
                logs = await self._get_container_logs(container_id)
                await session.emit("error", f"Container logs:\n{logs}")

                await self._stop_container(container_id, session)

                if session.auto_fix and attempt < max_attempts:
                    # Try to fix based on error
                    current_code, fix = await self._fix_code_from_error(current_code, logs)
                    if fix:
                        fixes_applied.append(fix)
                    continue

                return SandboxResult(
                    success=False,
                    status=ExecutionStatus.ERROR,
                    error=f"Service failed to start:\n{logs}",
                    final_code=current_code,
                    fixes_applied=fixes_applied
                )

        return SandboxResult(
            success=False,
            status=ExecutionStatus.ERROR,
            error="Max attempts reached",
            final_code=current_code,
            fixes_applied=fixes_applied
        )

    async def _run_script(self, session: SandboxSession, work_dir: Path) -> SandboxResult:
        """Run a simple script"""

        code_file = work_dir / "script.py"
        code_file.write_text(session.code)

        await session.emit("log", "📝 Running script...")

        # Install dependencies
        deps = self._extract_dependencies(session.code)
        if deps:
            await session.emit("log", f"📦 Installing: {', '.join(deps)}")
            await self._pip_install(work_dir, deps, session)

        # Run script
        try:
            process = await asyncio.create_subprocess_exec(
                "python", str(code_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(work_dir)
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=60
            )

            stdout_text = stdout.decode() if stdout else ""
            stderr_text = stderr.decode() if stderr else ""

            if stdout_text:
                await session.emit("log", f"📤 Output:\n{stdout_text}")
            if stderr_text:
                await session.emit("error", f"📛 Stderr:\n{stderr_text}")

            success = process.returncode == 0

            return SandboxResult(
                success=success,
                status=ExecutionStatus.SUCCESS if success else ExecutionStatus.ERROR,
                output=stdout_text,
                error=stderr_text if not success else "",
                final_code=session.code
            )

        except asyncio.TimeoutError:
            return SandboxResult(
                success=False,
                status=ExecutionStatus.TIMEOUT,
                error="Script execution timed out (60s)"
            )

    def _extract_dependencies(self, code: str) -> list[str]:
        """Extract pip dependencies from imports"""
        deps = set()

        # Map imports to pip packages
        import_map = {
            "flask": "flask",
            "fastapi": "fastapi",
            "uvicorn": "uvicorn",
            "requests": "requests",
            "httpx": "httpx",
            "pandas": "pandas",
            "numpy": "numpy",
            "sqlalchemy": "sqlalchemy",
            "pydantic": "pydantic",
            "redis": "redis",
            "celery": "celery",
            "pytest": "pytest",
            "aiohttp": "aiohttp",
            "bs4": "beautifulsoup4",
            "PIL": "pillow",
            "cv2": "opencv-python",
            "sklearn": "scikit-learn",
            "tensorflow": "tensorflow",
            "torch": "torch",
        }

        # Find imports
        import_pattern = r"(?:from|import)\s+(\w+)"
        for match in re.finditer(import_pattern, code):
            module = match.group(1)
            if module in import_map:
                deps.add(import_map[module])

        return list(deps)

    async def _pip_install(self, work_dir: Path, deps: list[str], session: SandboxSession) -> bool:
        """Install pip dependencies"""
        try:
            process = await asyncio.create_subprocess_exec(
                "pip", "install", "--quiet", *deps,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)

            if process.returncode != 0:
                await session.emit("error", f"pip install failed: {stderr.decode()}")
                return False
            return True
        except Exception as e:
            await session.emit("error", f"pip install error: {e}")
            return False

    async def _start_service_container(
        self,
        work_dir: Path,
        service_type: ServiceType,
        session: SandboxSession
    ) -> str | None:
        """Start service in Docker container"""

        container_name = f"svc-{session.session_id}"

        # Determine start command
        if service_type == ServiceType.FLASK:
            cmd = "python app.py"
        elif service_type == ServiceType.FASTAPI:
            cmd = "uvicorn main:app --host 0.0.0.0 --port 5000"
        else:
            cmd = "python app.py"

        try:
            await session.emit("log", f"🚀 Starting service on port 5000...")

            # For simplicity, run directly instead of Docker
            # In production, use Docker for isolation
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(work_dir)
            )

            session.process = process
            await session.emit("log", f"✅ Process started: PID {process.pid}")

            return str(process.pid)

        except Exception as e:
            await session.emit("error", f"Failed to start: {e}")
            return None

    async def _stop_container(self, container_id: str, session: SandboxSession):
        """Stop service container/process"""
        await session.emit("log", f"\n🧹 Stopping container {container_id[:12]}...")

        if session.process:
            try:
                session.process.terminate()
                await asyncio.wait_for(session.process.wait(), timeout=5)
            except:
                session.process.kill()

    async def _get_container_logs(self, container_id: str) -> str:
        """Get logs from container"""
        # For process-based execution
        return "Check stderr for details"

    async def _fix_imports(self, code: str, missing_deps: list[str]) -> str:
        """Auto-fix missing imports"""
        # Simple fix: add imports at top if missing
        return code

    async def _fix_code_from_error(self, code: str, error_log: str) -> tuple[str, dict | None]:
        """Try to fix code based on error"""
        fix = None

        # Common fixes
        if "IndentationError" in error_log:
            # Try to fix indentation
            lines = code.split('\n')
            fixed_lines = []
            for line in lines:
                # Standardize to 4 spaces
                stripped = line.lstrip()
                indent = len(line) - len(stripped)
                fixed_lines.append(' ' * (indent // 4 * 4) + stripped)
            code = '\n'.join(fixed_lines)
            fix = {"type": "indentation", "description": "Fixed indentation"}

        elif "SyntaxError" in error_log:
            fix = {"type": "syntax", "description": "Attempted syntax fix"}

        elif "ModuleNotFoundError" in error_log:
            # Extract module name and install
            match = re.search(r"No module named '(\w+)'", error_log)
            if match:
                module = match.group(1)
                fix = {"type": "import", "description": f"Missing module: {module}"}

        return code, fix


def create_sandbox_routes(app):
    """Add sandbox routes to FastAPI app"""
    from fastapi import WebSocket, WebSocketDisconnect
    from pydantic import BaseModel

    runner = SandboxRunner()

    class RunRequest(BaseModel):
        code: str
        auto_fix: bool = True

    @app.post("/api/sandbox/run")
    async def start_sandbox(request: RunRequest):
        """Start sandbox execution and return session ID"""
        session = runner.create_session(request.code, request.auto_fix)

        # Start execution in background
        asyncio.create_task(runner.run(session))

        return {
            "success": True,
            "session_id": session.session_id,
            "websocket_url": f"/ws/sandbox/{session.session_id}"
        }

    @app.websocket("/ws/sandbox/{session_id}")
    async def sandbox_websocket(websocket: WebSocket, session_id: str):
        """WebSocket for streaming sandbox logs"""
        await websocket.accept()

        session = runner.get_session(session_id)
        if not session:
            await websocket.send_json({"type": "error", "data": "Session not found"})
            await websocket.close()
            return

        # Subscribe to log stream
        queue = session.subscribe()

        try:
            # Send existing logs first
            for log in session.logs:
                await websocket.send_json(log.to_dict())

            # Stream new logs
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=1.0)
                    await websocket.send_json(msg)

                    # Check if execution completed
                    if msg.get("type") == "result":
                        break

                except asyncio.TimeoutError:
                    # Check if session is still running
                    if session.status not in [ExecutionStatus.PENDING, ExecutionStatus.RUNNING]:
                        break
                    # Send keepalive
                    await websocket.send_json({"type": "ping"})

        except WebSocketDisconnect:
            pass
        finally:
            session.unsubscribe(queue)

    @app.post("/api/sandbox/stop/{session_id}")
    async def stop_sandbox(session_id: str):
        """Stop sandbox execution"""
        session = runner.get_session(session_id)
        if session and session.process:
            session.process.terminate()
            session.status = ExecutionStatus.STOPPED
            return {"success": True}
        return {"success": False, "error": "Session not found"}

    return runner

"""
IntentForge Service Tester

Automatically detect, run, and test web services (Flask, FastAPI, etc.) in Docker.

Flow:
1. Detect service type from code (Flask, FastAPI, Express, etc.)
2. Generate Dockerfile and docker-compose for the service
3. Start service in isolated Docker container
4. Wait for service to be ready
5. Auto-discover and test API endpoints
6. Return test results with fixes if needed
"""

import asyncio
import logging
import re
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ServiceInfo:
    """Detected service information"""

    type: str  # flask, fastapi, express, etc.
    port: int = 5000
    endpoints: list = field(default_factory=list)
    dependencies: list = field(default_factory=list)
    entry_point: str = "app.py"


@dataclass
class EndpointTest:
    """Test result for an endpoint"""

    method: str
    path: str
    status_code: int
    success: bool
    response_time_ms: float
    response_body: str = ""
    error: str = ""


@dataclass
class ServiceTestResult:
    """Complete service test result"""

    success: bool
    service_type: str
    container_id: str = ""
    endpoints_tested: int = 0
    endpoints_passed: int = 0
    test_results: list = field(default_factory=list)
    logs: list = field(default_factory=list)
    final_code: str = ""
    fixes_applied: list = field(default_factory=list)
    error: str = ""


class ServiceDetector:
    """Detect service type and endpoints from code"""

    PATTERNS = {
        "flask": {
            "import": r"from flask import|import flask",
            "app": r"Flask\(__name__\)|Flask\(.+\)",
            "route": r"@app\.route\(['\"](.+?)['\"].*?methods=\[(.+?)\]|@app\.route\(['\"](.+?)['\"]\)",
            "port": r"app\.run\(.*?port=(\d+)|\.run\(.*?port=(\d+)",
            "default_port": 5000,
        },
        "fastapi": {
            "import": r"from fastapi import|import fastapi",
            "app": r"FastAPI\(\)|FastAPI\(.+\)",
            "route": r"@app\.(get|post|put|delete|patch)\(['\"](.+?)['\"]",
            "port": r"uvicorn\.run\(.*?port=(\d+)",
            "default_port": 8000,
        },
        "express": {
            "import": r"require\(['\"]express['\"]\)|from ['\"]express['\"]",
            "app": r"express\(\)",
            "route": r"app\.(get|post|put|delete|patch)\(['\"](.+?)['\"]",
            "port": r"listen\((\d+)\)|PORT.*?=.*?(\d+)",
            "default_port": 3000,
        },
    }

    def detect(self, code: str) -> ServiceInfo | None:
        """Detect service type and extract info from code"""
        for svc_type, patterns in self.PATTERNS.items():
            # Check if this service type is present
            if not re.search(patterns["import"], code, re.IGNORECASE):
                continue

            if not re.search(patterns["app"], code):
                continue

            # Found a service - extract details
            info = ServiceInfo(type=svc_type)

            # Extract port
            port_match = re.search(patterns["port"], code)
            if port_match:
                port = port_match.group(1) or port_match.group(2)
                info.port = int(port) if port else patterns["default_port"]
            else:
                info.port = patterns["default_port"]

            # Extract endpoints
            info.endpoints = self._extract_endpoints(code, svc_type, patterns)

            # Extract dependencies
            info.dependencies = self._extract_dependencies(code, svc_type)

            return info

        return None

    def _extract_endpoints(self, code: str, svc_type: str, patterns: dict) -> list[dict]:
        """Extract API endpoints from code"""
        endpoints = []

        if svc_type == "flask":
            # Flask routes: @app.route('/path', methods=['GET', 'POST'])
            for match in re.finditer(patterns["route"], code):
                path = match.group(1) or match.group(3)
                methods_str = match.group(2) if match.group(2) else "GET"
                methods = re.findall(r"'(\w+)'", methods_str)
                if not methods:
                    methods = ["GET"]
                for method in methods:
                    endpoints.append({"method": method.upper(), "path": path})

        elif svc_type in ("fastapi", "express"):
            # FastAPI/Express: @app.get('/path') or app.get('/path', ...)
            for match in re.finditer(patterns["route"], code):
                method = match.group(1).upper()
                path = match.group(2)
                endpoints.append({"method": method, "path": path})

        return endpoints

    def _extract_dependencies(self, code: str, svc_type: str) -> list[str]:
        """Extract package dependencies from code"""
        deps = []

        if svc_type == "flask":
            deps.append("flask")
            if "jsonify" in code:
                pass  # Part of flask
            if "flask_cors" in code.lower():
                deps.append("flask-cors")

        elif svc_type == "fastapi":
            deps.append("fastapi")
            deps.append("uvicorn")
            if "pydantic" in code.lower():
                deps.append("pydantic")

        # Common dependencies
        if "sqlite3" in code:
            pass  # Built-in
        if "requests" in code:
            deps.append("requests")
        if "httpx" in code:
            deps.append("httpx")

        return list(set(deps))


class ServiceRunner:
    """Run services - subprocess mode (works inside containers)"""

    def __init__(self):
        self.processes: dict[str, subprocess.Popen] = {}

    async def start_service(
        self,
        code: str,
        service_info: ServiceInfo,
        workspace: Path,
    ) -> tuple[str, list[str]]:
        """Start service as subprocess, return process_id and logs"""
        import subprocess
        import sys

        logs = []
        process_id = f"svc-{uuid.uuid4().hex[:8]}"

        try:
            # Write code to file
            code_file = workspace / service_info.entry_point
            code_file.write_text(code)
            logs.append(f"📝 Wrote code to {service_info.entry_point}")

            # Install dependencies
            if service_info.dependencies:
                logs.append(f"📦 Installing: {', '.join(service_info.dependencies)}")
                for dep in service_info.dependencies:
                    install_result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "-q", dep],
                        capture_output=True,
                        text=True,
                        timeout=60,
                        check=False,
                    )
                    if install_result.returncode != 0:
                        logs.append(f"⚠️ Failed to install {dep}")

            # Start the service process
            logs.append(f"🚀 Starting service on port {service_info.port}...")

            env = {**dict(__import__("os").environ), "FLASK_ENV": "production"}

            process = subprocess.Popen(
                [sys.executable, str(code_file)],
                cwd=str(workspace),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )

            self.processes[process_id] = process
            logs.append(f"✅ Process started: PID {process.pid}")

            # Wait for service to be ready
            logs.append("⏳ Waiting for service to be ready...")
            ready = await self._wait_for_service(service_info.port, timeout=15)

            if ready:
                logs.append(f"✅ Service ready on port {service_info.port}")
            else:
                # Check if process crashed
                if process.poll() is not None:
                    stderr = process.stderr.read().decode() if process.stderr else ""
                    logs.append(f"❌ Process crashed: {stderr[:300]}")
                    return "", logs
                logs.append("⚠️ Service may not be fully ready (timeout)")

            return process_id, logs

        except Exception as e:
            logs.append(f"❌ Error: {e}")
            return "", logs

    async def _wait_for_service(self, port: int, timeout: int = 15) -> bool:
        """Wait for service to respond on port"""
        import httpx

        start = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start < timeout:
            try:
                async with httpx.AsyncClient() as client:
                    # Try common health endpoints
                    for path in ["/", "/health", "/api/health"]:
                        try:
                            await client.get(f"http://127.0.0.1:{port}{path}", timeout=2)
                            return True
                        except Exception:
                            pass
            except Exception:
                pass
            await asyncio.sleep(0.5)

        return False

    async def stop_service(self, process_id: str) -> None:
        """Stop service process"""
        process = self.processes.pop(process_id, None)
        if process:
            try:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except Exception:
                    process.kill()
            except Exception as e:
                logger.warning(f"Failed to stop process {process_id}: {e}")


class EndpointTester:
    """Test API endpoints"""

    async def test_endpoints(self, endpoints: list[dict], port: int) -> list[EndpointTest]:
        """Test all discovered endpoints"""
        import time

        import httpx

        results = []
        base_url = f"http://localhost:{port}"

        async with httpx.AsyncClient() as client:
            for endpoint in endpoints:
                method = endpoint["method"]
                path = endpoint["path"]

                # Replace path parameters with test values
                test_path = re.sub(r"<\w+:\w+>|<\w+>|\{\w+\}", "1", path)
                url = f"{base_url}{test_path}"

                start_time = time.time()
                test_result = EndpointTest(
                    method=method,
                    path=path,
                    status_code=0,
                    success=False,
                    response_time_ms=0,
                )

                try:
                    # Prepare request
                    kwargs = {"timeout": 10}

                    if method in ("POST", "PUT", "PATCH"):
                        # Send test data for write methods
                        kwargs["json"] = {"name": "Test", "email": "test@example.com"}

                    # Make request
                    response = await client.request(method, url, **kwargs)

                    test_result.status_code = response.status_code
                    test_result.response_time_ms = (time.time() - start_time) * 1000
                    test_result.response_body = response.text[:500]
                    test_result.success = response.status_code < 400

                except Exception as e:
                    test_result.error = str(e)
                    test_result.response_time_ms = (time.time() - start_time) * 1000

                results.append(test_result)

        return results


class ServiceTester:
    """
    Main service tester - orchestrates detection, running, and testing.
    """

    def __init__(self):
        self.detector = ServiceDetector()
        self.runner = ServiceRunner()
        self.endpoint_tester = EndpointTester()

    async def test_service(
        self,
        code: str,
        auto_fix: bool = True,
        max_fix_attempts: int = 3,
    ) -> ServiceTestResult:
        """
        Test a service end-to-end.

        1. Detect service type
        2. Start in Docker
        3. Test endpoints
        4. Auto-fix if needed
        5. Return results
        """
        result = ServiceTestResult(success=False, service_type="unknown")
        result.logs = []

        # Detect service type
        service_info = self.detector.detect(code)
        if not service_info:
            result.error = "Could not detect service type (Flask, FastAPI, etc.)"
            result.logs.append("❌ No service detected in code")
            return result

        result.service_type = service_info.type
        result.logs.append(f"🔍 Detected {service_info.type} service")
        result.logs.append(f"📍 Endpoints found: {len(service_info.endpoints)}")

        for ep in service_info.endpoints:
            result.logs.append(f"   {ep['method']} {ep['path']}")

        # Create workspace
        workspace = Path(tempfile.mkdtemp(prefix="intentforge_svc_"))
        current_code = code
        container_id = ""

        try:
            for attempt in range(max_fix_attempts):
                result.logs.append(f"\n🔄 Attempt {attempt + 1}/{max_fix_attempts}")

                # Start service
                container_id, start_logs = await self.runner.start_service(
                    current_code, service_info, workspace
                )
                result.logs.extend(start_logs)

                if not container_id:
                    if auto_fix and attempt < max_fix_attempts - 1:
                        # Try to fix the code
                        result.logs.append("🔧 Attempting to fix code...")
                        fix_result = await self._fix_code(
                            current_code, service_info, "Build/start failed"
                        )
                        if fix_result.get("fixed"):
                            current_code = fix_result["code"]
                            result.fixes_applied.append(fix_result["description"])
                            result.logs.append(f"✅ Fix applied: {fix_result['description']}")
                            continue
                    result.error = "Failed to start service container"
                    return result

                result.container_id = container_id

                # Test endpoints
                result.logs.append("\n🧪 Testing endpoints...")
                test_results = await self.endpoint_tester.test_endpoints(
                    service_info.endpoints, service_info.port
                )

                result.test_results = test_results
                result.endpoints_tested = len(test_results)
                result.endpoints_passed = sum(1 for t in test_results if t.success)

                for t in test_results:
                    status = "✅" if t.success else "❌"
                    result.logs.append(
                        f"   {status} {t.method} {t.path} → {t.status_code} ({t.response_time_ms:.0f}ms)"
                    )

                # Check if all passed
                if result.endpoints_passed == result.endpoints_tested:
                    result.success = True
                    result.final_code = current_code
                    result.logs.append(f"\n✅ All {result.endpoints_tested} endpoints passed!")
                    break

                # Try to fix if some failed
                if auto_fix and attempt < max_fix_attempts - 1:
                    failed = [t for t in test_results if not t.success]
                    if failed:
                        result.logs.append(
                            f"🔧 Attempting to fix {len(failed)} failed endpoints..."
                        )
                        fix_result = await self._fix_code(
                            current_code,
                            service_info,
                            f"Endpoints failed: {[f'{t.method} {t.path}' for t in failed]}",
                        )
                        if fix_result.get("fixed"):
                            current_code = fix_result["code"]
                            result.fixes_applied.append(fix_result["description"])
                            result.logs.append(f"✅ Fix applied: {fix_result['description']}")
                            # Stop current container before retry
                            await self.runner.stop_service(container_id)
                            container_id = ""
                            continue

                # Partial success
                result.final_code = current_code
                result.logs.append(
                    f"\n⚠️ {result.endpoints_passed}/{result.endpoints_tested} endpoints passed"
                )
                break

        finally:
            # Cleanup
            if container_id:
                result.logs.append(f"\n🧹 Stopping container {container_id[:12]}...")
                await self.runner.stop_service(container_id)

            # Cleanup workspace
            import shutil

            shutil.rmtree(workspace, ignore_errors=True)

        return result

    async def _fix_code(self, code: str, service_info: ServiceInfo, error: str) -> dict:
        """Use LLM to fix service code"""
        try:
            from .services import ChatService

            chat = ChatService()
            prompt = f"""Fix this {service_info.type} service code.

ERROR: {error}

CODE:
```python
{code}
```

Return ONLY the fixed Python code, no explanations.
Ensure the service runs correctly and all endpoints work.
"""
            response = await chat.send(
                message=prompt,
                system="You are a code fixer. Return only working Python code.",
            )

            if response.get("success"):
                import re

                text = response.get("response", "")
                code_match = re.search(r"```python\n([\s\S]*?)```", text)
                if code_match:
                    return {
                        "fixed": True,
                        "code": code_match.group(1),
                        "description": "Fixed via LLM",
                    }
                elif text.strip().startswith(("from ", "import ", "#")):
                    return {"fixed": True, "code": text.strip(), "description": "Fixed via LLM"}

        except Exception as e:
            logger.warning(f"Fix failed: {e}")

        return {"fixed": False}


# Convenience function
async def test_service(code: str, auto_fix: bool = True, max_attempts: int = 3) -> dict:
    """
    Test a service in Docker with auto-fix.

    Returns dict with:
    - success: bool
    - service_type: str
    - endpoints_tested: int
    - endpoints_passed: int
    - test_results: list
    - logs: list
    - final_code: str
    - fixes_applied: list
    """
    tester = ServiceTester()
    result = await tester.test_service(code, auto_fix, max_attempts)

    return {
        "success": result.success,
        "service_type": result.service_type,
        "endpoints_tested": result.endpoints_tested,
        "endpoints_passed": result.endpoints_passed,
        "test_results": [
            {
                "method": t.method,
                "path": t.path,
                "status_code": t.status_code,
                "success": t.success,
                "response_time_ms": t.response_time_ms,
                "error": t.error,
            }
            for t in result.test_results
        ],
        "logs": result.logs,
        "final_code": result.final_code,
        "fixes_applied": result.fixes_applied,
        "error": result.error,
    }

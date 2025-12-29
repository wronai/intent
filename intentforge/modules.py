"""
IntentForge Module Manager

Autonomous module creation, management, and execution system.
Enables LLM to create, deploy, and use reusable service modules.

Each module is:
- A complete, runnable service with Dockerfile
- Self-contained with dependencies
- Accessible via DSL commands
- Stored in modules/[name]/ directory
"""

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

logger = logging.getLogger(__name__)

# Base paths
MODULES_DIR = Path(__file__).parent.parent / "modules"
TEMPLATE_DIR = MODULES_DIR / ".template"


@dataclass
class ModuleInfo:
    """Module metadata"""

    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = "IntentForge"
    port: int = 8080
    status: str = "created"  # created, building, running, stopped, error
    container_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    endpoints: list = field(default_factory=list)


@dataclass
class ModuleExecutionResult:
    """Result of module execution"""

    success: bool
    module: str
    action: str
    result: Any = None
    error: str = ""
    execution_time_ms: float = 0


class ModuleManager:
    """
    Manages autonomous service modules.

    Features:
    - Create new modules from templates or LLM-generated code
    - Build and run modules as Docker containers
    - Execute module actions via HTTP
    - Chain multiple module calls
    - Register modules as DSL services
    """

    def __init__(self):
        self.modules: dict[str, ModuleInfo] = {}
        self.base_port = 9100  # Starting port for modules
        self._ensure_directories()
        self._load_existing_modules()

    def _ensure_directories(self):
        """Ensure required directories exist"""
        MODULES_DIR.mkdir(parents=True, exist_ok=True)

    def _load_existing_modules(self):
        """Load existing modules from disk"""
        for module_dir in MODULES_DIR.iterdir():
            if module_dir.is_dir() and not module_dir.name.startswith("."):
                config_file = module_dir / "module.yaml"
                if config_file.exists():
                    try:
                        with open(config_file) as f:
                            config = yaml.safe_load(f)
                        self.modules[config["name"]] = ModuleInfo(
                            name=config["name"],
                            version=config.get("version", "1.0.0"),
                            description=config.get("description", ""),
                            port=config.get("config", {}).get("port", self._next_port()),
                            status="stopped",
                        )
                    except Exception as e:
                        logger.warning(f"Failed to load module {module_dir.name}: {e}")

    def _next_port(self) -> int:
        """Get next available port for a module"""
        used_ports = {m.port for m in self.modules.values()}
        port = self.base_port
        while port in used_ports:
            port += 1
        return port

    def list_modules(self) -> list[dict]:
        """List all available modules"""
        return [
            {
                "name": m.name,
                "version": m.version,
                "description": m.description,
                "status": m.status,
                "port": m.port,
            }
            for m in self.modules.values()
        ]

    async def create_module(
        self,
        name: str,
        description: str = "",
        code: str | None = None,
        requirements: list[str] | None = None,
        from_template: bool = True,
    ) -> ModuleInfo:
        """
        Create a new module.

        Args:
            name: Module name (lowercase, no spaces)
            description: Module description
            code: Main module code (optional, uses template if not provided)
            requirements: Additional Python packages
            from_template: Whether to copy from template

        Returns:
            ModuleInfo for the created module
        """
        name = name.lower().replace(" ", "_").replace("-", "_")
        module_dir = MODULES_DIR / name

        if module_dir.exists():
            raise ValueError(f"Module {name} already exists")

        # Copy template
        if from_template and TEMPLATE_DIR.exists():
            shutil.copytree(TEMPLATE_DIR, module_dir)
        else:
            module_dir.mkdir(parents=True)

        # Assign port
        port = self._next_port()

        # Update module.yaml
        config = {
            "name": name,
            "version": "1.0.0",
            "description": description,
            "author": "IntentForge",
            "config": {
                "port": port,
                "timeout": 30,
                "max_retries": 3,
            },
            "created_at": datetime.now().isoformat(),
        }

        with open(module_dir / "module.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Update main.py if custom code provided
        if code:
            main_py = module_dir / "main.py"
            # Wrap code in module template
            module_code = self._wrap_code_as_module(name, code, port)
            main_py.write_text(module_code)

        # Update requirements.txt
        if requirements:
            req_file = module_dir / "requirements.txt"
            existing = req_file.read_text() if req_file.exists() else ""
            new_reqs = "\n".join(requirements)
            req_file.write_text(existing + "\n" + new_reqs)

        # Update Dockerfile with correct port
        dockerfile = module_dir / "Dockerfile"
        if dockerfile.exists():
            content = dockerfile.read_text()
            content = content.replace("ENV MODULE_PORT=8080", f"ENV MODULE_PORT={port}")
            content = content.replace("EXPOSE 8080", f"EXPOSE {port}")
            dockerfile.write_text(content)

        # Create module info
        module_info = ModuleInfo(
            name=name,
            version="1.0.0",
            description=description,
            port=port,
            status="created",
        )
        self.modules[name] = module_info

        logger.info(f"Created module: {name} on port {port}")
        return module_info

    def _wrap_code_as_module(self, name: str, code: str, port: int) -> str:
        """Wrap user code as a complete module"""
        return f'''#!/usr/bin/env python3
"""
IntentForge Module: {name}
Auto-generated autonomous service module
"""

import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

MODULE_NAME = "{name}"
MODULE_VERSION = "1.0.0"
MODULE_PORT = {port}

app = FastAPI(title=f"IntentForge Module: {{MODULE_NAME}}")


@app.get("/health")
async def health():
    return {{"status": "healthy", "module": MODULE_NAME}}


@app.get("/info")
async def info():
    return {{"name": MODULE_NAME, "version": MODULE_VERSION, "endpoints": ["/health", "/info", "/execute"]}}


# ============================================================================
# USER CODE START
# ============================================================================

{code}

# ============================================================================
# USER CODE END
# ============================================================================


@app.post("/execute")
async def execute(request: Request):
    """Main execution endpoint"""
    body = await request.json()

    try:
        # Call the main process function
        if 'process' in dir():
            result = await process(body) if asyncio.iscoroutinefunction(process) else process(body)
        elif 'main' in dir():
            result = await main(body) if asyncio.iscoroutinefunction(main) else main(body)
        else:
            result = {{"message": "No process() or main() function defined"}}

        return JSONResponse(content={{"success": True, "module": MODULE_NAME, "result": result}})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={{"success": False, "module": MODULE_NAME, "error": str(e)}}
        )


import asyncio

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=MODULE_PORT)
'''

    async def build_module(self, name: str) -> bool:
        """Build module Docker image"""
        if name not in self.modules:
            raise ValueError(f"Module {name} not found")

        module_dir = MODULES_DIR / name
        if not module_dir.exists():
            raise ValueError(f"Module directory not found: {module_dir}")

        self.modules[name].status = "building"

        try:
            result = subprocess.run(
                ["docker", "build", "-t", f"intentforge-module-{name}", "."],
                check=False,
                cwd=str(module_dir),
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                self.modules[name].status = "built"
                logger.info(f"Built module: {name}")
                return True
            else:
                self.modules[name].status = "error"
                logger.error(f"Build failed for {name}: {result.stderr}")
                return False
        except Exception as e:
            self.modules[name].status = "error"
            logger.error(f"Build error for {name}: {e}")
            return False

    async def start_module(self, name: str) -> bool:
        """Start module as Docker container"""
        if name not in self.modules:
            raise ValueError(f"Module {name} not found")

        module = self.modules[name]

        # Stop if already running
        await self.stop_module(name)

        try:
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    f"intentforge-module-{name}",
                    "-p",
                    f"{module.port}:{module.port}",
                    "--network",
                    "intent_intentforge-network",
                    f"intentforge-module-{name}",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                module.container_id = result.stdout.strip()
                module.status = "running"
                logger.info(f"Started module: {name} on port {module.port}")
                return True
            else:
                module.status = "error"
                logger.error(f"Start failed for {name}: {result.stderr}")
                return False
        except Exception as e:
            module.status = "error"
            logger.error(f"Start error for {name}: {e}")
            return False

    async def stop_module(self, name: str) -> bool:
        """Stop module container"""
        if name not in self.modules:
            return False

        try:
            subprocess.run(
                ["docker", "stop", f"intentforge-module-{name}"],
                check=False,
                capture_output=True,
                timeout=30,
            )
            subprocess.run(
                ["docker", "rm", f"intentforge-module-{name}"],
                check=False,
                capture_output=True,
                timeout=30,
            )
            self.modules[name].status = "stopped"
            self.modules[name].container_id = ""
            return True
        except Exception:
            return False

    async def execute_module(
        self,
        name: str,
        action: str = "execute",
        data: dict | None = None,
        timeout: float = 30.0,
    ) -> ModuleExecutionResult:
        """
        Execute an action on a module.

        Args:
            name: Module name
            action: Action/endpoint to call (default: execute)
            data: Data to send to the module
            timeout: Request timeout in seconds

        Returns:
            ModuleExecutionResult
        """
        import time

        start_time = time.time()

        if name not in self.modules:
            return ModuleExecutionResult(
                success=False,
                module=name,
                action=action,
                error=f"Module {name} not found",
            )

        module = self.modules[name]

        if module.status != "running":
            return ModuleExecutionResult(
                success=False,
                module=name,
                action=action,
                error=f"Module {name} is not running (status: {module.status})",
            )

        url = f"http://localhost:{module.port}/{action}"

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=data or {})
                result = response.json()

                return ModuleExecutionResult(
                    success=result.get("success", response.status_code == 200),
                    module=name,
                    action=action,
                    result=result.get("result", result),
                    execution_time_ms=(time.time() - start_time) * 1000,
                )
        except Exception as e:
            return ModuleExecutionResult(
                success=False,
                module=name,
                action=action,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def create_from_llm(
        self,
        task_description: str,
        module_name: str | None = None,
    ) -> ModuleInfo:
        """
        Create a module from LLM-generated code.

        Args:
            task_description: Description of what the module should do
            module_name: Optional module name (auto-generated if not provided)

        Returns:
            Created ModuleInfo
        """
        from .services import ChatService

        chat = ChatService()

        prompt = f"""Create a Python module for the following task:

{task_description}

Requirements:
1. Create a function called `process(data: dict) -> dict` that handles the main logic
2. The function receives a dict with input data and returns a dict with results
3. Use only standard library or common packages (requests, httpx, json, etc.)
4. Include proper error handling
5. Return ONLY the Python code, no explanations

Example structure:
```python
def process(data: dict) -> dict:
    # Your implementation here
    result = ...
    return {{"success": True, "result": result}}
```
"""

        response = await chat.send(
            message=prompt,
            system="You are a Python code generator. Generate clean, functional code only. No explanations.",
        )

        if not response.get("success"):
            raise ValueError(f"LLM generation failed: {response.get('error')}")

        code = response.get("response", "")

        # Extract code from markdown if present
        import re

        code_match = re.search(r"```python\n([\s\S]*?)```", code)
        if code_match:
            code = code_match.group(1)

        # Generate module name if not provided
        if not module_name:
            name_prompt = f"Generate a short, lowercase, underscore-separated module name for: {task_description[:100]}"
            name_response = await chat.send(
                message=name_prompt, system="Reply with just the module name, nothing else."
            )
            module_name = name_response.get("response", "").strip().lower().replace(" ", "_")[:30]
            if not module_name:
                module_name = f"module_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Extract required packages from code
        requirements = []
        import_pattern = r"^import (\w+)|^from (\w+)"
        for line in code.split("\n"):
            match = re.match(import_pattern, line)
            if match:
                pkg = match.group(1) or match.group(2)
                if pkg not in ["os", "sys", "json", "re", "datetime", "typing", "asyncio"]:
                    requirements.append(pkg)

        # Create the module
        return await self.create_module(
            name=module_name,
            description=task_description,
            code=code,
            requirements=list(set(requirements)),
        )


# Global instance
module_manager = ModuleManager()

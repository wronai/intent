"""
IntentForge AutoDev Server

Integrated server providing:
- Chat with LLM (streaming)
- Sandbox execution with live logs
- Agent task execution
- WebSocket streaming for all components
"""

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# Import core components
try:
    from intentforge.sandbox_runner import SandboxRunner, create_sandbox_routes
    from intentforge.autoworker_agent import WorkerOrchestrator, create_agent_routes
except ImportError:
    # Fallback for local testing if not installed as package
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from intentforge.sandbox_runner import SandboxRunner, create_sandbox_routes
    from intentforge.autoworker_agent import WorkerOrchestrator, create_agent_routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Server configuration"""
    # LLM
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-coder:7b-instruct")
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Server
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))

    # Paths
    # Use repo root/workspace for sandbox
    WORKSPACE = Path(os.getenv("WORKSPACE", "/tmp/autodev"))
    STATIC_PATH = Path(__file__).parent / "usecases"


config = Config()
config.WORKSPACE.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Request/Response Models
# =============================================================================

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None
    system: Optional[str] = None
    action: Optional[str] = None


class CodeExecuteRequest(BaseModel):
    code: str
    language: str = "python"
    timeout: int = 60


class SandboxRunRequest(BaseModel):
    code: str
    auto_fix: bool = True


class AgentTaskRequest(BaseModel):
    task: str
    code: str = ""


# =============================================================================
# LLM Client
# =============================================================================

class OllamaClient:
    """Simple Ollama client for chat"""

    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host

    async def chat_stream(self, message: str, model: str, system: str = None):
        """Stream chat response"""
        import httpx

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST",
                    f"{self.host}/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": True
                    }
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if "message" in data and "content" in data["message"]:
                                    yield data["message"]["content"]
                                if data.get("done"):
                                    break
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            yield f"Error connecting to LLM: {str(e)}"

    async def chat(self, message: str, model: str, system: str = None) -> str:
        """Non-streaming chat"""
        import httpx

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{self.host}/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False
                    }
                )
                data = response.json()
                return data.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return f"Error connecting to LLM: {str(e)}"

    async def list_models(self) -> list[str]:
        """List available models"""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.host}/api/tags")
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except:
            return []


# =============================================================================
# Initialize Components
# =============================================================================

llm_client = OllamaClient(config.OLLAMA_HOST)
sandbox_runner = SandboxRunner(str(config.WORKSPACE / "sandbox"))
orchestrator = WorkerOrchestrator()


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="IntentForge AutoDev",
    description="AI Development Assistant with Auto-Sandbox",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Sandbox Routes
create_sandbox_routes(app)
# Register Agent Routes
app.include_router(create_agent_routes(orchestrator))


# =============================================================================
# Chat Endpoints
# =============================================================================

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat endpoint (non-streaming)"""

    # Handle special actions
    if request.action == "models":
        models = await llm_client.list_models()
        return {
            "success": True,
            "models": models,
            "provider": config.LLM_PROVIDER,
            "default_model": config.LLM_MODEL
        }

    # Regular chat
    model = request.model or config.LLM_MODEL

    try:
        start = time.time()
        response = await llm_client.chat(
            request.message,
            model,
            request.system
        )
        latency = (time.time() - start) * 1000

        return {
            "success": True,
            "response": response,
            "model": model,
            "latency_ms": latency
        }
    except Exception as e:
        logger.exception("Chat error")
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint"""

    model = request.model or config.LLM_MODEL

    async def generate():
        try:
            async for chunk in llm_client.chat_stream(
                request.message,
                model,
                request.system
            ):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


# =============================================================================
# Code Execution Endpoints
# =============================================================================

@app.post("/api/code/save")
async def save_code(request: dict):
    """Save code to file"""

    code = request.get("code", "")
    language = request.get("language", "python")

    ext_map = {
        "python": ".py",
        "javascript": ".js",
        "typescript": ".ts",
        "bash": ".sh",
        "html": ".html",
        "css": ".css",
        "json": ".json",
        "yaml": ".yaml"
    }

    ext = ext_map.get(language, ".txt")
    timestamp = int(time.time())
    filename = f"code_{timestamp}{ext}"

    # Save to workspace
    filepath = config.WORKSPACE / "saved" / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(code)

    return {
        "success": True,
        "filename": filename,
        "path": str(filepath)
    }


# =============================================================================
# Agent Endpoints
# =============================================================================

@app.post("/api/agent/task")
async def create_agent_task(request: AgentTaskRequest):
    """Create and start agent task"""

    # Classify task
    task_lower = request.task.lower()
    if any(w in task_lower for w in ["test", "verify"]):
        role = "qa"
    elif any(w in task_lower for w in ["deploy", "docker", "ci"]):
        role = "devops"
    else:
        role = "developer"

    task = orchestrator.create_task(role, request.task, request.code)

    # Start execution
    asyncio.create_task(execute_agent_task(task))

    return {
        "success": True,
        "task_id": task.id,
        "worker": task.role.value,
        "websocket_url": f"/ws/agent/{task.id}"
    }


async def execute_agent_task(task):
    """Execute agent task using LLM and Sandbox"""

    await orchestrator.emit_status(task.id, "running")

    try:
        await orchestrator.emit_log(task.id, f"🚀 Starting {task.role.value} worker...", "info")
        await orchestrator.emit_log(task.id, f"📋 Task: {task.description[:100]}...", "info")

        # Use LLM to generate response
        await orchestrator.emit_log(task.id, "🧠 Analyzing task...", "info")

        prompt = f"""You are an expert {task.role.value}. Complete this task:

{task.description}

{"Input code:\n```python\n" + task.code + "\n```" if task.code else ""}

Provide your solution with code if applicable. Enclose code in ```python blocks."""

        # Stream response
        await orchestrator.emit_log(task.id, "💭 Generating solution...", "info")

        response = ""
        # Mock streaming for now or use real client if integrated
        # Here we use the actual LLM client
        response = await llm_client.chat(prompt, config.LLM_MODEL)

        await orchestrator.emit_log(task.id, "✅ Solution generated", "success")

        # Extract code if present
        code_match = re.search(r'```(?:python)?\n([\s\S]*?)```', response)
        code = code_match.group(1).strip() if code_match else ""

        if code:
             await orchestrator.emit_log(task.id, "💾 Code extracted, validating in sandbox...", "info")
             # Validate in sandbox
             session = sandbox_runner.create_session(code, auto_fix=False)
             result = await sandbox_runner.run(session)

             if result.success:
                 await orchestrator.emit_log(task.id, "✅ Sandbox validation passed", "success")
             else:
                 await orchestrator.emit_log(task.id, f"⚠️ Sandbox validation failed: {result.error}", "warning")

        await orchestrator.emit_result(task.id, response)

    except Exception as e:
        await orchestrator.emit_log(task.id, f"❌ Error: {e}", "error")
        await orchestrator.emit_status(task.id, "failed")


# =============================================================================
# Static Files
# =============================================================================

@app.get("/")
async def index():
    """Serve main page"""
    html_path = config.STATIC_PATH / "autodev_chat.html"
    if html_path.exists():
        return FileResponse(html_path)
    return {"message": "IntentForge AutoDev API. Please ensure autodev_chat.html exists in examples/usecases/", "docs": "/docs"}


# Mount static files if directory exists
if config.STATIC_PATH.exists():
    app.mount("/static", StaticFiles(directory=str(config.STATIC_PATH)), name="static")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting IntentForge AutoDev Server on {config.HOST}:{config.PORT}")
    logger.info(f"LLM: {config.LLM_PROVIDER} / {config.LLM_MODEL}")

    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info"
    )

import asyncio
import contextlib
import inspect
import os
import subprocess
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

# Add parent to path to import intentforge
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from intentforge import Intent, IntentForge, IntentType, TargetPlatform
from intentforge.services import services

# Initialize FastAPI
app = FastAPI(title="IntentForge API Server")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Rate Limiting
# =============================================================================

RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))  # requests per window
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # window in seconds

# In-memory rate limit storage (use Redis in production)
rate_limit_data: dict[str, list[float]] = defaultdict(list)


def get_client_ip(request: Request) -> str:
    """Get client IP from request, considering proxies"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rate_limit(client_ip: str) -> tuple[bool, dict]:
    """Check if client is within rate limit. Returns (allowed, info)"""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    # Clean old entries
    rate_limit_data[client_ip] = [t for t in rate_limit_data[client_ip] if t > window_start]

    current_requests = len(rate_limit_data[client_ip])
    remaining = max(0, RATE_LIMIT_REQUESTS - current_requests)
    reset_time = int(window_start + RATE_LIMIT_WINDOW)

    info = {
        "X-RateLimit-Limit": str(RATE_LIMIT_REQUESTS),
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Reset": str(reset_time),
    }

    if current_requests >= RATE_LIMIT_REQUESTS:
        return False, info

    # Record this request
    rate_limit_data[client_ip].append(now)
    return True, info


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    if not RATE_LIMIT_ENABLED:
        return await call_next(request)

    # Skip rate limiting for static files
    if request.url.path.startswith("/examples/") or request.url.path.startswith("/static/"):
        return await call_next(request)

    client_ip = get_client_ip(request)
    allowed, info = check_rate_limit(client_ip)

    if not allowed:
        return JSONResponse(
            status_code=429,
            content={
                "success": False,
                "error": "Rate limit exceeded",
                "retry_after": RATE_LIMIT_WINDOW,
            },
            headers=info,
        )

    response = await call_next(request)
    # Add rate limit headers to response
    for key, value in info.items():
        response.headers[key] = value
    return response


# =============================================================================
# API Key Authentication
# =============================================================================

API_KEYS_ENABLED = os.getenv("API_KEYS_ENABLED", "false").lower() == "true"
API_KEYS = set(filter(None, os.getenv("API_KEYS", "").split(",")))

# Public endpoints that don't require authentication
PUBLIC_ENDPOINTS = {"/health", "/docs", "/openapi.json", "/redoc"}


async def verify_api_key(request: Request) -> bool:
    """Verify API key from header or query parameter"""
    if not API_KEYS_ENABLED:
        return True

    # Check if endpoint is public
    if request.url.path in PUBLIC_ENDPOINTS:
        return True

    # Skip auth for static files
    if request.url.path.startswith("/examples/") or request.url.path.startswith("/static/"):
        return True

    # Get API key from header or query
    api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")

    if not api_key:
        return False

    return api_key in API_KEYS


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Middleware to check API key authentication"""
    if not await verify_api_key(request):
        return JSONResponse(
            status_code=401,
            content={
                "success": False,
                "error": "Invalid or missing API key",
                "hint": "Set X-API-Key header or api_key query parameter",
            },
        )
    return await call_next(request)


# Initialize IntentForge
# We use Ollama by default as per dev workflow
provider = os.getenv("LLM_PROVIDER", "ollama")
model = os.getenv("LLM_MODEL", "llama3.1:8b")

print(f"Initializing IntentForge with Provider: {provider}, Model: {model}")

forge = IntentForge(
    enable_auto_deploy=True,  # We want to execute the code
    sandbox_mode=True,  # Safely
    provider=provider,
    model=model,
)


class IntentRequest(BaseModel):
    description: str
    intent_type: str = "workflow"  # Default to workflow/generic
    context: dict[str, Any] = {}


class IntentResponse(BaseModel):
    success: bool
    message: str
    result: Any | None = None
    original_intent: str


@app.post("/api/intent", response_model=IntentResponse)
async def process_intent(request: IntentRequest):
    """
    Process a natural language intent
    """
    print(f"Received intent: {request.description}")

    # Map string type to Enum
    try:
        if request.intent_type == "workflow":
            i_type = IntentType.WORKFLOW
        elif request.intent_type == "api":
            i_type = IntentType.API_ENDPOINT
        else:
            i_type = IntentType.WORKFLOW

        # Create Intent
        intent = Intent(
            description=request.description,
            intent_type=i_type,
            target_platform=TargetPlatform.GENERIC_PYTHON,
            context=request.context,
        )

        # Process
        result = await forge.process_intent(intent)

        if not result.success:
            return IntentResponse(
                success=False,
                message=f"Generation failed: {result.validation_errors}",
                original_intent=request.description,
            )

        # Even if generation succeeded, execution might fail or be empty if auto_deploy is off
        # However, we enabled auto_deploy=True.
        # The result.execution_result should contain the return value of the executed code

        exec_res = result.execution_result

        if exec_res and not exec_res.success:
            print("❌ Execution Failed")
            print("FAILED CODE:")
            print("---")
            print(result.generated_code)
            print("---")
            print(f"Error: {exec_res.error}")

        return IntentResponse(
            success=True,
            message="Intent processed successfully",
            result=exec_res,
            original_intent=request.description,
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


# =============================================================================
# Streaming Chat Endpoint
# =============================================================================


@app.post("/api/chat/stream")
async def chat_stream(request: Request):
    """
    Streaming chat endpoint - returns Server-Sent Events (SSE)

    Example:
        POST /api/chat/stream {"message": "Hello", "model": "llama3.1:8b"}
    """
    from intentforge.llm.providers import LLMConfig, get_llm_provider

    body = await request.json()
    message = body.get("message", "")
    model = body.get("model")
    system = body.get("system", "Jesteś pomocnym asystentem AI. Odpowiadaj po polsku.")

    async def generate():
        import json as json_lib

        try:
            config = LLMConfig.from_env()
            if model:
                config.model = model
            provider = get_llm_provider(config=config)

            async for chunk in provider.generate_stream(message, system=system):
                # SSE format
                yield f"data: {json_lib.dumps({'chunk': chunk})}\n\n"

            yield f"data: {json_lib.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json_lib.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# WebSocket Streaming Endpoint
# =============================================================================


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for streaming chat responses.

    Client sends: {"message": "Hello", "model": "llama3.1:8b", "system": "..."}
    Server streams: {"chunk": "..."} or {"done": true} or {"error": "..."}
    """
    await websocket.accept()

    try:
        from intentforge.llm.providers import LLMConfig, get_llm_provider

        while True:
            # Receive message from client
            data = await websocket.receive_json()

            message = data.get("message", "")
            model = data.get("model")
            system = data.get("system", "Jesteś pomocnym asystentem AI. Odpowiadaj po polsku.")

            if not message:
                await websocket.send_json({"error": "No message provided"})
                continue

            try:
                config = LLMConfig.from_env()
                if model:
                    config.model = model
                provider = get_llm_provider(config=config)

                # Stream response
                async for chunk in provider.generate_stream(message, system=system):
                    await websocket.send_json({"chunk": chunk})

                await websocket.send_json({"done": True})

            except Exception as e:
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        with contextlib.suppress(Exception):
            await websocket.send_json({"error": str(e)})


# =============================================================================
# WebSocket: Sandbox Streaming
# =============================================================================

# Active sandbox sessions for streaming
sandbox_sessions: dict[str, list[WebSocket]] = {}


@app.websocket("/ws/sandbox/{session_id}")
async def websocket_sandbox(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for streaming sandbox execution logs.

    Client connects to /ws/sandbox/{session_id}
    Server streams: {"type": "log|status|result|error", "data": "..."}
    """
    await websocket.accept()

    # Register this websocket for the session
    if session_id not in sandbox_sessions:
        sandbox_sessions[session_id] = []
    sandbox_sessions[session_id].append(websocket)

    try:
        # Keep connection alive and handle client messages
        while True:
            try:
                data = await websocket.receive_json()
                # Client can send commands like {"action": "stop"}
                if data.get("action") == "stop":
                    await broadcast_sandbox_log(session_id, "🛑 Stop requested by client", "status")
            except Exception:
                await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    finally:
        # Cleanup
        if session_id in sandbox_sessions:
            sandbox_sessions[session_id] = [
                ws for ws in sandbox_sessions[session_id] if ws != websocket
            ]
            if not sandbox_sessions[session_id]:
                del sandbox_sessions[session_id]


async def broadcast_sandbox_log(session_id: str, message: str, msg_type: str = "log"):
    """Broadcast log message to all connected clients for a session"""
    if session_id not in sandbox_sessions:
        return

    dead_sockets = []
    for ws in sandbox_sessions[session_id]:
        try:
            await ws.send_json({"type": msg_type, "data": message, "timestamp": time.time()})
        except Exception:
            dead_sockets.append(ws)

    # Cleanup dead sockets
    for ws in dead_sockets:
        sandbox_sessions[session_id].remove(ws)


@app.post("/api/agent/run")
async def agent_run_endpoint(request: Request):
    """
    Run autonomous agent on a task.

    The agent will:
    1. Analyze task and context
    2. Find/reuse existing modules
    3. Generate code if needed
    4. Build, test, and register new modules
    5. Make autonomous decisions

    POST /api/agent/run {
        "task": "Create a REST API for users",
        "code": ""  // optional existing code
    }
    """
    from intentforge.agent import run_agent

    body = await request.json()
    task = body.get("task", "")
    code = body.get("code", "")

    if not task:
        return JSONResponse(
            status_code=400, content={"success": False, "error": "No task provided"}
        )

    try:
        result = await run_agent(task, code)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/api/agent/modules")
async def list_agent_modules():
    """List all modules built by the agent"""
    from intentforge.agent import AutonomousAgent

    agent = AutonomousAgent()
    modules = agent.list_modules()

    return JSONResponse(
        content={
            "success": True,
            "modules": [
                {
                    "id": m.id,
                    "name": m.name,
                    "description": m.description,
                    "version": m.version,
                    "status": m.status.value,
                    "tests": f"{m.tests_passed}/{m.tests_total}",
                    "use_count": m.use_count,
                    "tags": m.tags,
                }
                for m in modules
            ],
        }
    )


@app.post("/api/sandbox/run")
async def sandbox_run_streaming(request: Request):
    """
    Run code in sandbox with WebSocket streaming.

    POST /api/sandbox/run {
        "code": "from flask import Flask...",
        "session_id": "abc123",  // optional, for WebSocket streaming
        "auto_fix": true
    }

    Returns immediately with session_id, streams logs via WebSocket.
    """
    import asyncio
    import uuid

    body = await request.json()
    code = body.get("code", "")
    session_id = body.get("session_id") or str(uuid.uuid4())[:8]
    auto_fix = body.get("auto_fix", True)

    if not code:
        return JSONResponse(
            status_code=400, content={"success": False, "error": "No code provided"}
        )

    # Start sandbox execution in background
    async def run_sandbox():
        from intentforge.service_tester import ServiceDetector, ServiceTester

        detector = ServiceDetector()
        service_info = detector.detect(code)

        if service_info:
            await broadcast_sandbox_log(
                session_id, f"🔍 Detected {service_info.type} service", "status"
            )
            await broadcast_sandbox_log(
                session_id, f"📍 Found {len(service_info.endpoints)} endpoints", "log"
            )
            for ep in service_info.endpoints:
                await broadcast_sandbox_log(session_id, f"   {ep['method']} {ep['path']}", "log")

            # Run full service test with streaming
            tester = ServiceTester()
            result = await tester.test_service(code, auto_fix=auto_fix)

            # Stream all logs
            for log in result.logs:
                await broadcast_sandbox_log(session_id, log, "log")
                await asyncio.sleep(0.05)  # Small delay for visual effect

            # Send final result
            await broadcast_sandbox_log(
                session_id,
                {
                    "success": result.success,
                    "service_type": result.service_type,
                    "endpoints_tested": result.endpoints_tested,
                    "endpoints_passed": result.endpoints_passed,
                    "final_code": result.final_code,
                },
                "result",
            )
        else:
            # Regular code execution
            await broadcast_sandbox_log(session_id, "🚀 Running code...", "status")

            from intentforge.code_runner import run_with_autofix

            result = await run_with_autofix(code, auto_fix=auto_fix)

            for log in result.get("logs", []):
                await broadcast_sandbox_log(session_id, log, "log")
                await asyncio.sleep(0.05)

            await broadcast_sandbox_log(
                session_id,
                {
                    "success": result.get("success"),
                    "output": result.get("output"),
                    "error": result.get("error"),
                    "final_code": result.get("final_code"),
                },
                "result",
            )

    # Run in background
    _task = asyncio.create_task(run_sandbox())  # noqa: RUF006

    return JSONResponse(
        content={
            "success": True,
            "session_id": session_id,
            "message": "Sandbox started, connect to WebSocket for logs",
            "websocket_url": f"/ws/sandbox/{session_id}",
        }
    )


# =============================================================================
# Proactive Processing Endpoint
# =============================================================================


@app.post("/api/proactive/process")
async def proactive_process(request: Request):
    """
    Proactive processing endpoint with intelligent decision-making.

    Automatically:
    - Detects content type and applies best processing strategy
    - Retries with fallback methods if initial processing fails
    - Extracts structured data from documents
    - Executes and debugs generated code

    POST /api/proactive/process {
        "content": "...",  // base64 image, text, or code
        "type": "image|text|code|document",
        "options": {"auto_execute": true, "extract_data": true}
    }
    """
    from intentforge.proactive import ProactiveEngine, ProcessingContext
    from intentforge.services import FileService

    body = await request.json()
    content = body.get("content", "")
    content_type = body.get("type", "auto")
    options = body.get("options", {})

    # Auto-detect content type
    if content_type == "auto":
        if content.startswith("/9j/") or content.startswith("iVBOR"):
            content_type = "image"
        elif (
            "```" in content
            or content.strip().startswith("def ")
            or content.strip().startswith("function ")
        ):
            content_type = "code"
        else:
            content_type = "text"

    # Create processing context
    context = ProcessingContext(
        input_type=content_type,
        content=content,
        metadata=options,
    )

    # For images, run initial OCR and Vision analysis
    if content_type == "image":
        file_service = FileService()

        # Run Vision analysis first
        vision_result = await file_service._analyze_image_with_vision(
            content,
            prompt="Przeanalizuj ten obraz. Opisz co widzisz, wykryj obiekty, "
            "rozpoznaj tekst (OCR) jeśli jest widoczny. Odpowiedz po polsku.",
        )

        context.results["vision"] = vision_result
        context.metadata["vision_detected_text"] = bool(
            vision_result.get("success")
            and any(
                word in vision_result.get("response", "").lower()
                for word in [
                    "tekst",
                    "napis",
                    "słow",
                    "dokument",
                    "faktur",
                    "paragon",
                    "data",
                    "numer",
                ]
            )
        )

        # Run Tesseract OCR
        ocr_result = await file_service.ocr(image_base64=content, use_tesseract=True)
        context.results["ocr"] = ocr_result
        context.metadata["ocr_failed"] = (
            not ocr_result.get("success") or not ocr_result.get("text", "").strip()
        )

        # If OCR failed but Vision detected text, try Vision-based OCR
        if context.metadata["ocr_failed"] and context.metadata["vision_detected_text"]:
            vision_ocr = await file_service._analyze_image_with_vision(
                content,
                prompt="Przepisz DOKŁADNIE i KOMPLETNIE cały tekst widoczny na tym obrazie. "
                "Zachowaj oryginalny układ tekstu (linie, kolumny). "
                "NIE dodawaj żadnych komentarzy ani opisów - TYLKO tekst z obrazu. "
                "Jeśli tekst jest w tabeli, zachowaj strukturę tabeli.",
            )

            if vision_ocr.get("success"):
                ocr_text = vision_ocr.get("response", "")
                # Filter out "no text" responses
                if "brak tekstu" not in ocr_text.lower() and len(ocr_text) > 20:
                    context.results["ocr"] = {
                        "success": True,
                        "text": ocr_text,
                        "method": "vision_fallback",
                        "model": file_service.vision_model,
                    }
                    context.metadata["ocr_failed"] = False

    # Run proactive engine
    engine = ProactiveEngine()
    result = await engine.process(context)

    # Combine all results
    return JSONResponse(
        content={
            "success": True,
            "content_type": content_type,
            "processing": result,
            "ocr": context.results.get("ocr", {}),
            "vision": context.results.get("vision", {}),
            "extracted_data": context.results.get("extract_data", {}),
        }
    )


@app.post("/api/proactive/code")
async def proactive_code(request: Request):
    """
    Proactive code execution with automatic debugging.

    POST /api/proactive/code {
        "code": "print('hello')",
        "language": "python",
        "auto_debug": true
    }
    """
    from intentforge.proactive import Decision, DecisionType, ProactiveEngine, ProcessingContext

    body = await request.json()
    code = body.get("code", "")
    language = body.get("language", "python")
    auto_debug = body.get("auto_debug", True)

    engine = ProactiveEngine()

    # Execute code
    exec_decision = Decision(
        type=DecisionType.EXECUTE_CODE,
        action="run",
        params={"code": code, "language": language},
    )

    context = ProcessingContext(input_type="code", content=code)
    exec_result = await engine._handle_execute_code(exec_decision, context)

    result = {
        "success": exec_result.get("success", False),
        "execution": exec_result,
    }

    # If execution failed and auto_debug is enabled, debug the code
    if not exec_result.get("success") and auto_debug:
        debug_decision = Decision(
            type=DecisionType.DEBUG_CODE,
            action="analyze",
            params={
                "code": code,
                "language": language,
                "error": exec_result.get("stderr", "") or exec_result.get("error", ""),
            },
        )

        debug_result = await engine._handle_debug_code(debug_decision, context)
        result["debug"] = debug_result

    return JSONResponse(content=result)


# =============================================================================
# Service API Endpoints (LLM-powered)
# =============================================================================


def _filter_kwargs(fn, data: dict[str, Any]) -> dict[str, Any]:
    """Filter kwargs to only include parameters accepted by the function"""
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return data
    allowed = set(sig.parameters)
    return {k: v for k, v in data.items() if k in allowed}


async def _call_service(service: str, action: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Call a service method with the given payload"""
    svc = services.get(service)
    if svc is None:
        raise HTTPException(status_code=404, detail=f"Unknown service: {service}")

    method = getattr(svc, action, None)
    if method is None:
        raise HTTPException(
            status_code=404, detail=f"Unknown action '{action}' for service '{service}'"
        )

    kwargs = payload.copy()
    kwargs.pop("action", None)
    kwargs.pop("request_id", None)

    try:
        result = method(**_filter_kwargs(method, kwargs))
        if inspect.isawaitable(result):
            result = await result
    except TypeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if isinstance(result, dict):
        return result

    return {"success": True, "result": result}


@app.post("/api/{service}")
async def api_service(service: str, request: Request) -> JSONResponse:
    """
    Generic service endpoint - routes to appropriate service handler.

    Services: chat, analytics, voice, file, form, payment, camera, data, email

    Example:
        POST /api/chat {"action": "send", "message": "Hello"}
        POST /api/analytics {"action": "stats", "period": "current_month"}
        POST /api/voice {"action": "process", "command": "Turn on lights"}
    """
    body = await request.json()
    action = body.get("action")
    if not action or not isinstance(action, str):
        return JSONResponse(
            status_code=400, content={"success": False, "error": "Missing 'action' parameter"}
        )

    request_id = body.get("request_id")

    try:
        result = await _call_service(service, action, body)
        if request_id is not None and isinstance(result, dict):
            result = {**result, "request_id": request_id}
        return JSONResponse(content=result)
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code, content={"success": False, "error": e.detail}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# =============================================================================
# Code Execution Endpoint
# =============================================================================

# Language execution templates
LANGUAGE_TEMPLATES = {
    "python": {
        "extension": ".py",
        "command": ["python3", "{file}"],
        "docker": "python:3.11-slim",
        "setup": None,
    },
    "javascript": {
        "extension": ".js",
        "command": ["node", "{file}"],
        "docker": "node:20-slim",
        "setup": None,
    },
    "typescript": {
        "extension": ".ts",
        "command": ["npx", "ts-node", "{file}"],
        "docker": "node:20-slim",
        "setup": "npm install -g ts-node typescript",
    },
    "bash": {
        "extension": ".sh",
        "command": ["bash", "{file}"],
        "docker": "alpine:latest",
        "setup": None,
    },
    "shell": {
        "extension": ".sh",
        "command": ["sh", "{file}"],
        "docker": "alpine:latest",
        "setup": None,
    },
    "ruby": {
        "extension": ".rb",
        "command": ["ruby", "{file}"],
        "docker": "ruby:3.2-slim",
        "setup": None,
    },
    "php": {
        "extension": ".php",
        "command": ["php", "{file}"],
        "docker": "php:8.2-cli",
        "setup": None,
    },
    "go": {
        "extension": ".go",
        "command": ["go", "run", "{file}"],
        "docker": "golang:1.21-alpine",
        "setup": None,
    },
    "rust": {
        "extension": ".rs",
        "command": ["rustc", "{file}", "-o", "{output}", "&&", "{output}"],
        "docker": "rust:1.74-slim",
        "setup": None,
    },
    "java": {
        "extension": ".java",
        "command": ["java", "{file}"],
        "docker": "openjdk:21-slim",
        "setup": None,
    },
    "c": {
        "extension": ".c",
        "command": ["gcc", "{file}", "-o", "{output}", "&&", "{output}"],
        "docker": "gcc:13",
        "setup": None,
    },
    "cpp": {
        "extension": ".cpp",
        "command": ["g++", "{file}", "-o", "{output}", "&&", "{output}"],
        "docker": "gcc:13",
        "setup": None,
    },
    "sql": {
        "extension": ".sql",
        "command": ["sqlite3", ":memory:", "-init", "{file}"],
        "docker": None,
        "setup": None,
    },
    "html": {
        "extension": ".html",
        "command": None,  # Open in browser
        "docker": None,
        "setup": None,
        "serve": True,
    },
}

# Directory for generated code files
CODE_OUTPUT_DIR = Path("/tmp/intentforge_code")
CODE_OUTPUT_DIR.mkdir(exist_ok=True)


@app.post("/api/code/save")
async def save_code(request: Request):
    """
    Save code to a file and optionally execute it.

    POST /api/code/save {
        "code": "print('hello')",
        "language": "python",
        "filename": "optional_name.py",
        "execute": true
    }
    """
    body = await request.json()
    code = body.get("code", "")
    language = body.get("language", "python").lower()
    filename = body.get("filename")
    execute = body.get("execute", False)

    if not code:
        return JSONResponse(
            status_code=400, content={"success": False, "error": "No code provided"}
        )

    # Get language template
    template = LANGUAGE_TEMPLATES.get(language)
    if not template:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": f"Unsupported language: {language}",
                "supported": list(LANGUAGE_TEMPLATES.keys()),
            },
        )

    # Generate filename if not provided
    if not filename:
        file_id = str(uuid.uuid4())[:8]
        filename = f"code_{file_id}{template['extension']}"

    # Ensure correct extension
    if not filename.endswith(template["extension"]):
        filename += template["extension"]

    # Save file
    filepath = CODE_OUTPUT_DIR / filename
    filepath.write_text(code, encoding="utf-8")

    result = {
        "success": True,
        "filepath": str(filepath),
        "filename": filename,
        "language": language,
        "size": len(code),
    }

    # Execute if requested
    if execute and template.get("command"):
        exec_result = await execute_code_file(str(filepath), language)
        result["execution"] = exec_result

    return JSONResponse(content=result)


@app.post("/api/code/execute")
async def execute_code(request: Request):
    """
    Execute code directly without saving.

    POST /api/code/execute {
        "code": "print('hello')",
        "language": "python",
        "timeout": 30
    }
    """
    body = await request.json()
    code = body.get("code", "")
    language = body.get("language", "python").lower()
    timeout = min(body.get("timeout", 30), 60)  # Max 60 seconds

    if not code:
        return JSONResponse(
            status_code=400, content={"success": False, "error": "No code provided"}
        )

    template = LANGUAGE_TEMPLATES.get(language)
    if not template or not template.get("command"):
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": f"Cannot execute language: {language}",
                "supported": [k for k, v in LANGUAGE_TEMPLATES.items() if v.get("command")],
            },
        )

    # Create temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=template["extension"], delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        temp_path = f.name

    try:
        exec_result = await execute_code_file(temp_path, language, timeout)
        return JSONResponse(content={"success": True, **exec_result})
    finally:
        # Cleanup temp file
        with contextlib.suppress(OSError):
            os.unlink(temp_path)


# =============================================================================
# Autonomous Module API
# =============================================================================


@app.get("/api/modules")
async def list_modules():
    """List all available modules"""
    from intentforge.modules import module_manager

    return JSONResponse(
        content={
            "success": True,
            "modules": module_manager.list_modules(),
        }
    )


@app.post("/api/modules/create")
async def create_module(request: Request):
    """
    Create a new autonomous module.

    POST /api/modules/create {
        "name": "my_module",
        "description": "Module description",
        "code": "def process(data): return {'result': data}",
        "requirements": ["requests"]
    }
    """
    from intentforge.modules import module_manager

    body = await request.json()
    name = body.get("name")
    description = body.get("description", "")
    code = body.get("code")
    requirements = body.get("requirements", [])

    if not name:
        return JSONResponse(
            status_code=400, content={"success": False, "error": "Module name required"}
        )

    try:
        module_info = await module_manager.create_module(
            name=name,
            description=description,
            code=code,
            requirements=requirements,
        )

        return JSONResponse(
            content={
                "success": True,
                "module": {
                    "name": module_info.name,
                    "version": module_info.version,
                    "port": module_info.port,
                    "status": module_info.status,
                },
            }
        )
    except Exception as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})


@app.post("/api/modules/create-from-task")
async def create_module_from_task(request: Request):
    """
    Create a module from natural language task description.
    LLM generates the code automatically.

    POST /api/modules/create-from-task {
        "task": "Create a CSV parser that extracts column headers and data",
        "name": "csv_parser"  // optional
    }
    """
    from intentforge.modules import module_manager

    body = await request.json()
    task = body.get("task", "")
    name = body.get("name")

    if not task:
        return JSONResponse(
            status_code=400, content={"success": False, "error": "Task description required"}
        )

    try:
        module_info = await module_manager.create_from_llm(
            task_description=task,
            module_name=name,
        )

        return JSONResponse(
            content={
                "success": True,
                "module": {
                    "name": module_info.name,
                    "version": module_info.version,
                    "port": module_info.port,
                    "status": module_info.status,
                    "description": module_info.description,
                },
            }
        )
    except Exception as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})


@app.post("/api/modules/{module_name}/build")
async def build_module(module_name: str):
    """Build module Docker image"""
    from intentforge.modules import module_manager

    try:
        success = await module_manager.build_module(module_name)
        return JSONResponse(content={"success": success, "module": module_name})
    except Exception as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})


@app.post("/api/modules/{module_name}/start")
async def start_module(module_name: str):
    """Start module container"""
    from intentforge.modules import module_manager

    try:
        success = await module_manager.start_module(module_name)
        module = module_manager.modules.get(module_name)
        return JSONResponse(
            content={
                "success": success,
                "module": module_name,
                "port": module.port if module else None,
            }
        )
    except Exception as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})


@app.post("/api/modules/{module_name}/stop")
async def stop_module(module_name: str):
    """Stop module container"""
    from intentforge.modules import module_manager

    try:
        success = await module_manager.stop_module(module_name)
        return JSONResponse(content={"success": success, "module": module_name})
    except Exception as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})


@app.post("/api/modules/{module_name}/execute")
async def execute_module(module_name: str, request: Request):
    """
    Execute a module action.

    POST /api/modules/{module_name}/execute {
        "action": "execute",
        "data": {"input": "value"}
    }
    """
    from intentforge.modules import module_manager

    body = await request.json()
    action = body.get("action", "execute")
    data = body.get("data", {})

    try:
        result = await module_manager.execute_module(
            name=module_name,
            action=action,
            data=data,
        )

        return JSONResponse(
            content={
                "success": result.success,
                "module": module_name,
                "result": result.result,
                "error": result.error,
                "execution_time_ms": result.execution_time_ms,
            }
        )
    except Exception as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})


# =============================================================================
# Autonomous Agent API
# =============================================================================


@app.post("/api/autonomous/execute")
async def autonomous_execute(request: Request):
    """
    Execute a complex task autonomously.

    The agent will:
    1. Plan the task into steps
    2. Execute each step (LLM, code, modules)
    3. Self-correct on failures
    4. Create reusable modules if needed

    POST /api/autonomous/execute {
        "task": "Create a web scraper that extracts product prices from a URL",
        "context": {"url": "https://example.com"},
        "max_steps": 10
    }
    """
    from intentforge.autonomous import autonomous_agent

    body = await request.json()
    task = body.get("task", "")
    context = body.get("context", {})
    max_steps = min(body.get("max_steps", 10), 20)

    if not task:
        return JSONResponse(
            status_code=400, content={"success": False, "error": "Task description required"}
        )

    try:
        result = await autonomous_agent.execute_task(
            task=task,
            context=context,
            max_steps=max_steps,
        )

        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/api/autonomous/history")
async def autonomous_history():
    """Get task execution history"""
    from intentforge.autonomous import autonomous_agent

    return JSONResponse(
        content={
            "success": True,
            "history": autonomous_agent.task_history[-20:],  # Last 20 tasks
        }
    )


@app.post("/api/code/run-service")
async def run_service_endpoint(request: Request):
    """
    Run and test a web service (Flask, FastAPI, etc.) in Docker.

    Flow:
    1. Detect service type (Flask, FastAPI, Express)
    2. Build Docker image with dependencies
    3. Start container and wait for ready
    4. Auto-discover and test all endpoints
    5. Auto-fix if tests fail
    6. Return detailed test results

    POST /api/code/run-service {
        "code": "from flask import Flask...",
        "auto_fix": true,
        "max_attempts": 3
    }

    Response:
    {
        "success": true,
        "service_type": "flask",
        "endpoints_tested": 5,
        "endpoints_passed": 5,
        "test_results": [...],
        "logs": [...],
        "final_code": "..."
    }
    """
    from intentforge.service_tester import test_service

    body = await request.json()
    code = body.get("code", "")
    auto_fix = body.get("auto_fix", True)
    max_attempts = min(body.get("max_attempts", 3), 5)

    if not code:
        return JSONResponse(
            status_code=400, content={"success": False, "error": "No code provided"}
        )

    try:
        result = await test_service(
            code=code,
            auto_fix=auto_fix,
            max_attempts=max_attempts,
        )
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/code/git-tracked")
async def git_tracked_execution(request: Request):
    """
    Execute code with Git-based iteration tracking.

    Each fix attempt is committed to Git, providing:
    - Full history of changes (git log)
    - Diffs between iterations (git diff)
    - Context from history for better LLM fixes
    - Pattern analysis from previous fixes

    POST /api/code/git-tracked {
        "code": "print(undefined_var)",
        "intent": "Print a variable",
        "max_iterations": 5
    }

    Response:
    {
        "success": true,
        "final_code": "...",
        "iterations": 3,
        "commits": ["abc123", "def456", "ghi789"],
        "git_context": "## Recent Iteration History..."
    }
    """
    from intentforge.code_runner import run_with_autofix
    from intentforge.git_manager import git_tracker

    body = await request.json()
    code = body.get("code", "")
    intent = body.get("intent", "Execute code")
    max_iterations = min(body.get("max_iterations", 5), 10)

    if not code:
        return JSONResponse(
            status_code=400, content={"success": False, "error": "No code provided"}
        )

    try:
        # Start Git session
        branch = await git_tracker.start_session(code, intent)
        commits = []
        current_code = code
        success = False

        for _iteration in range(max_iterations):
            # Execute code
            result = await run_with_autofix(
                code=current_code,
                language="python",
                max_retries=1,
                auto_install=True,
            )

            # Record iteration
            commit = await git_tracker.record_iteration(
                code=current_code,
                error=result.get("error") if not result.get("success") else None,
                fix_description=result.get("fixes", [{}])[0].get("description")
                if result.get("fixes")
                else None,
                success=result.get("success", False),
            )

            if commit:
                commits.append(commit.hash)

            if result.get("success"):
                success = True
                break

            # Get context from Git history for next iteration
            git_context = await git_tracker.get_context_for_llm()

            # Use LLM with Git context to fix
            from intentforge.services import ChatService

            chat = ChatService()

            fix_response = await chat.send(
                message=f"""Fix this code based on the error and iteration history:

CODE:
```python
{current_code}
```

ERROR:
{result.get("error", "Unknown error")}

{git_context}

Return ONLY the fixed Python code, no explanations.
""",
                system="You are a code fixer. Use the iteration history to avoid repeating failed approaches. Return only code.",
            )

            if fix_response.get("success"):
                import re

                response_text = fix_response.get("response", "")
                code_match = re.search(r"```python\n([\s\S]*?)```", response_text)
                if code_match:
                    current_code = code_match.group(1)
                elif response_text.strip():
                    current_code = response_text.strip()

        # End session
        session_result = await git_tracker.end_session(merge=success, squash=True)

        return JSONResponse(
            content={
                "success": success,
                "final_code": current_code,
                "iterations": session_result.get("total_iterations", 0),
                "commits": commits,
                "branch": branch,
                "git_context": await git_tracker.get_context_for_llm() if not success else None,
            }
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/code/auto-conversation")
async def auto_conversation_endpoint(request: Request):
    """
    Execute code with automatic conversation branching for error resolution.

    Flow:
    1. Execute code
    2. If errors detected, spawn sub-conversations for each problem
    3. Each sub-conversation uses LLM to analyze and fix
    4. Results merge back, continue until resolved

    POST /api/code/auto-conversation {
        "code": "print(undefined_var)",
        "intent": "Print a variable",
        "auto_branch": true
    }

    Response:
    {
        "success": true/false,
        "final_code": "...",
        "threads": 3,
        "problems_resolved": 2,
        "conversation_tree": {...}
    }
    """
    from intentforge.conversation_engine import process_with_auto_conversation

    body = await request.json()
    code = body.get("code", "")
    intent = body.get("intent", "")
    auto_branch = body.get("auto_branch", True)

    if not code:
        return JSONResponse(
            status_code=400, content={"success": False, "error": "No code provided"}
        )

    try:
        result = await process_with_auto_conversation(
            code=code,
            intent=intent,
            auto_branch=auto_branch,
        )
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/code/test-and-fix")
async def test_and_fix_code_endpoint(request: Request):
    """
    Test code with auto-generated tests and fix until all tests pass.

    Flow:
    1. Generate tests from code intent (or use provided tests)
    2. Run tests against code
    3. If tests fail, analyze and fix code via LLM
    4. Repeat until all tests pass or max iterations

    POST /api/code/test-and-fix {
        "code": "def add(a, b): return a + b",
        "intent": "Add two numbers and return the sum",
        "tests": "def test_add(): assert add(2, 3) == 5",  // optional
        "language": "python",
        "max_iterations": 5
    }

    Response:
    {
        "success": true/false,
        "final_code": "...",
        "tests": "...",
        "iterations": 2,
        "test_results": {"total": 5, "passed": 5, "failed": 0},
        "logs": ["🧪 Starting...", "✅ All tests passed!"]
    }
    """
    from intentforge.code_tester import test_and_fix_code

    body = await request.json()
    code = body.get("code", "")
    intent = body.get("intent", "")
    tests = body.get("tests")
    language = body.get("language", "python").lower()
    max_iterations = min(body.get("max_iterations", 5), 10)

    if not code:
        return JSONResponse(
            status_code=400, content={"success": False, "error": "No code provided"}
        )

    if not intent:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Intent/description required"},
        )

    try:
        result = await test_and_fix_code(
            code=code,
            intent=intent,
            tests=tests,
            language=language,
            max_iterations=max_iterations,
        )

        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/code/autofix")
async def execute_code_autofix(request: Request):
    """
    Execute code with automatic debugging, fixing, and retry.

    When code fails:
    1. Detects error type (missing module, syntax, runtime)
    2. Attempts to fix (install deps, fix via LLM)
    3. Retries execution
    4. Loops until success or max retries

    POST /api/code/autofix {
        "code": "import requests\\nprint(requests.get('...'))",
        "language": "python",
        "max_retries": 3,
        "auto_install": true
    }

    Response:
    {
        "success": true/false,
        "output": "...",
        "attempts": 2,
        "fixes": [{"type": "install_package", "description": "Installed requests"}],
        "final_code": "..."
    }
    """
    from intentforge.code_runner import run_with_autofix

    body = await request.json()
    code = body.get("code", "")
    language = body.get("language", "python").lower()
    max_retries = min(body.get("max_retries", 3), 5)  # Max 5 retries
    auto_install = body.get("auto_install", True)

    if not code:
        return JSONResponse(
            status_code=400, content={"success": False, "error": "No code provided"}
        )

    try:
        result = await run_with_autofix(
            code=code,
            language=language,
            max_retries=max_retries,
            auto_install=auto_install,
        )

        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


async def execute_code_file(filepath: str, language: str, timeout: int = 30) -> dict:
    """Execute a code file and return results"""
    template = LANGUAGE_TEMPLATES.get(language)
    if not template or not template.get("command"):
        return {"error": f"Cannot execute {language}"}

    # Build command
    output_path = filepath.replace(template["extension"], "")
    cmd = []
    for part in template["command"]:
        cmd.append(part.format(file=filepath, output=output_path))

    # Handle compound commands (with &&)
    if "&&" in cmd:
        cmd_str = " ".join(cmd)
        cmd = ["sh", "-c", cmd_str]

    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(CODE_OUTPUT_DIR),
        )

        return {
            "stdout": result.stdout[:10000],  # Limit output
            "stderr": result.stderr[:5000],
            "returncode": result.returncode,
            "success": result.returncode == 0,
            "command": " ".join(cmd),
        }
    except subprocess.TimeoutExpired:
        return {
            "error": f"Execution timed out after {timeout}s",
            "success": False,
        }
    except FileNotFoundError as e:
        return {
            "error": f"Command not found: {e}",
            "success": False,
        }
    except Exception as e:
        return {
            "error": str(e),
            "success": False,
        }


@app.get("/api/code/templates")
async def get_templates():
    """Get available language templates"""
    return JSONResponse(
        content={
            "success": True,
            "templates": {
                lang: {
                    "extension": t["extension"],
                    "executable": t.get("command") is not None,
                    "docker": t.get("docker"),
                }
                for lang, t in LANGUAGE_TEMPLATES.items()
            },
        }
    )


@app.get("/api/code/files")
async def list_code_files():
    """List saved code files"""
    files = []
    for f in CODE_OUTPUT_DIR.iterdir():
        if f.is_file():
            files.append(
                {
                    "name": f.name,
                    "path": str(f),
                    "size": f.stat().st_size,
                    "modified": f.stat().st_mtime,
                }
            )
    return JSONResponse(content={"success": True, "files": files})


# Serve static files for the frontend example
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8085"))
    uvicorn.run(app, host="0.0.0.0", port=port)

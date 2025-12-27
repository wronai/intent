import inspect
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .core import Intent, IntentForge
from .broker import MQTTIntentBroker
from .schema_registry import SchemaType, get_registry
from .services import services


settings = get_settings()


def _filter_kwargs(fn, data: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return data
    allowed = {name for name in sig.parameters.keys()}
    return {k: v for k, v in data.items() if k in allowed}


async def _call_service(service: str, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    svc = services.get(service)
    if svc is None:
        raise HTTPException(status_code=404, detail=f"Unknown service: {service}")

    method = getattr(svc, action, None)
    if method is None:
        raise HTTPException(status_code=404, detail=f"Unknown action '{action}' for service '{service}'")

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


app = FastAPI(title=settings.app_name, version=settings.app_version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


_forge: Optional[IntentForge] = None
_broker: Optional[MQTTIntentBroker] = None


@app.on_event("startup")
async def _startup() -> None:
    global _forge, _broker

    _forge = IntentForge(
        mqtt_broker=settings.mqtt.host,
        mqtt_port=settings.mqtt.port,
        api_key=settings.llm.api_key.get_secret_value(),
        enable_auto_deploy=False,
        sandbox_mode=True,
    )

    _broker = MQTTIntentBroker(host=settings.mqtt.host, port=settings.mqtt.port, forge=_forge)
    if _broker.connect():
        _broker.start_listening()

    uploads_dir = os.getenv("UPLOAD_PATH", "./uploads")
    try:
        os.makedirs(uploads_dir, exist_ok=True)
        app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")
    except Exception:
        pass


@app.on_event("shutdown")
async def _shutdown() -> None:
    global _broker
    if _broker is not None:
        _broker.stop()
        _broker = None


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/generate")
async def api_generate(request: Request) -> JSONResponse:
    body = await request.json()

    schema_result = get_registry().validate(body, SchemaType.INTENT_REQUEST)
    if not schema_result.is_valid:
        return JSONResponse(status_code=400, content={"success": False, "errors": schema_result.errors})

    request_id = body.get("request_id")

    try:
        intent = Intent.from_dict(body)
    except Exception as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})

    if _forge is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    result = await _forge.process_intent(intent)
    payload: Dict[str, Any] = result.to_dict()

    if request_id is not None:
        payload["request_id"] = request_id

    return JSONResponse(content=payload)


@app.post("/api/{service}")
async def api_service(service: str, request: Request) -> JSONResponse:
    body = await request.json()
    action = body.get("action")
    if not action or not isinstance(action, str):
        return JSONResponse(status_code=400, content={"success": False, "error": "Missing 'action'"})

    request_id = body.get("request_id")

    result = await _call_service(service, action, body)
    if request_id is not None and isinstance(result, dict):
        result = {**result, "request_id": request_id}

    return JSONResponse(content=result)


def run() -> None:
    import uvicorn

    uvicorn.run("intentforge.server:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    run()

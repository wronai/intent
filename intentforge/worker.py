import asyncio
import contextlib
import inspect
import json
import logging
import os
import signal
from typing import Any

import paho.mqtt.client as mqtt

from .config import get_settings
from .core import Intent, IntentForge
from .services import services

logger = logging.getLogger(__name__)
settings = get_settings()


def _filter_kwargs(fn, data: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return data
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in data.items() if k in allowed}


async def _handle_intent_request(
    forge: IntentForge,
    mqtt_client: mqtt.Client,
    client_id: str,
    request_id: str | None,
    payload: dict[str, Any],
) -> None:
    logger.info("[INTENT] Processing intent request_id=%s from client=%s", request_id, client_id)
    logger.info("[INTENT] Payload: %s", json.dumps(payload, indent=2)[:500])
    try:
        intent = Intent.from_dict(payload)
        logger.info(
            "[INTENT] Created Intent: type=%s, platform=%s",
            intent.intent_type,
            intent.target_platform,
        )
        result = await forge.process_intent(intent)
        logger.info("[INTENT] Processing complete: success=%s", result.success)
        if not result.success:
            logger.warning("[INTENT] Validation errors: %s", result.validation_errors)
        response: dict[str, Any] = result.to_dict()
        if request_id is not None:
            response["request_id"] = request_id
        logger.info("[INTENT] Sending response to topic: intentforge/intent/response/%s", client_id)
        mqtt_client.publish(
            f"intentforge/intent/response/{client_id}",
            json.dumps(response),
            qos=1,
        )
    except Exception as e:
        logger.error("[INTENT] Failed: %s", str(e), exc_info=True)
        mqtt_client.publish(
            f"intentforge/intent/response/{client_id}",
            json.dumps({"success": False, "error": str(e), "request_id": request_id}),
            qos=1,
        )


async def _handle_action_request(
    forge: IntentForge,
    mqtt_client: mqtt.Client,
    action: str,
    client_id: str,
    request_id: str | None,
    payload: dict[str, Any],
) -> None:
    response_topic = f"intentforge/{action}/response/{client_id}"

    if action == "generate":
        try:
            intent = Intent.from_dict(payload)
            result = await forge.process_intent(intent)
            response: dict[str, Any] = result.to_dict()
            if request_id is not None:
                response["request_id"] = request_id
            mqtt_client.publish(response_topic, json.dumps(response), qos=1)
            return
        except Exception as e:
            mqtt_client.publish(
                response_topic,
                json.dumps({"success": False, "error": str(e), "request_id": request_id}),
                qos=1,
            )
            return

    try:
        svc = services.get(action)
        if svc is None:
            mqtt_client.publish(
                response_topic,
                json.dumps(
                    {
                        "success": False,
                        "error": f"Unknown service: {action}",
                        "request_id": request_id,
                    }
                ),
                qos=1,
            )
            return

        service_action = payload.get("action")
        if not service_action or not isinstance(service_action, str):
            mqtt_client.publish(
                response_topic,
                json.dumps(
                    {"success": False, "error": "Missing 'action'", "request_id": request_id}
                ),
                qos=1,
            )
            return

        method = getattr(svc, service_action, None)
        if method is None:
            mqtt_client.publish(
                response_topic,
                json.dumps(
                    {
                        "success": False,
                        "error": f"Unknown action '{service_action}' for service '{action}'",
                        "request_id": request_id,
                    }
                ),
                qos=1,
            )
            return

        kwargs = payload.copy()
        kwargs.pop("action", None)
        kwargs.pop("request_id", None)

        result = method(**_filter_kwargs(method, kwargs))
        if inspect.isawaitable(result):
            result = await result

        if isinstance(result, dict):
            response = result
            response.setdefault("success", True)
        else:
            response = {"success": True, "result": result}

        if request_id is not None:
            response["request_id"] = request_id

        mqtt_client.publish(response_topic, json.dumps(response), qos=1)

    except Exception as e:
        mqtt_client.publish(
            response_topic,
            json.dumps({"success": False, "error": str(e), "request_id": request_id}),
            qos=1,
        )


async def _run() -> None:
    logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO))

    forge = IntentForge(
        mqtt_broker=settings.mqtt.host,
        mqtt_port=settings.mqtt.port,
        api_key=settings.llm.api_key.get_secret_value(),
        enable_auto_deploy=False,
        sandbox_mode=True,
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _stop() -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _stop)

    mqtt_client = mqtt.Client(
        client_id=os.getenv("INTENTFORGE_WORKER_ID", "intentforge-worker"),
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    )

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code != 0:
            logger.error("MQTT connect failed rc=%s", reason_code)
            return
        logger.info("Connected to MQTT broker %s:%s", settings.mqtt.host, settings.mqtt.port)
        client.subscribe("intentforge/+/request/+", qos=1)

    def on_message(client, userdata, msg):
        topic = msg.topic
        logger.info("[MQTT] Received message on topic: %s", topic)
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            logger.info("[MQTT] Payload size: %d bytes", len(msg.payload))
        except Exception as e:
            logger.warning("[MQTT] Invalid JSON on topic %s: %s", topic, e)
            return

        parts = topic.split("/")
        if len(parts) < 4:
            return

        prefix, action, direction, client_id = parts[0], parts[1], parts[2], parts[3]
        if prefix != "intentforge" or direction != "request":
            return

        request_id = payload.get("request_id")
        logger.info("[MQTT] Action=%s, client_id=%s, request_id=%s", action, client_id, request_id)

        if action == "intent":
            logger.info("[MQTT] Dispatching to _handle_intent_request")
            asyncio.run_coroutine_threadsafe(
                _handle_intent_request(forge, client, client_id, request_id, payload),
                loop,
            )
        else:
            logger.info("[MQTT] Dispatching to _handle_action_request: action=%s", action)
            asyncio.run_coroutine_threadsafe(
                _handle_action_request(forge, client, action, client_id, request_id, payload),
                loop,
            )

    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    mqtt_client.connect(settings.mqtt.host, settings.mqtt.port, keepalive=60)
    mqtt_client.loop_start()

    await stop_event.wait()

    mqtt_client.loop_stop()
    mqtt_client.disconnect()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()

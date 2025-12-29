"""
IntentForge Service Handlers
Backend handlers for: forms, payments, email, camera, data
All use .env for configuration
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# Load environment
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration from .env
# =============================================================================


@dataclass
class ServiceConfig:
    """Configuration loaded from .env"""

    # Email (SMTP)
    smtp_host: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    smtp_from: str = os.getenv("SMTP_FROM", "")

    # PayPal
    paypal_client_id: str = os.getenv("PAYPAL_CLIENT_ID", "")
    paypal_secret: str = os.getenv("PAYPAL_SECRET", "")
    paypal_mode: str = os.getenv("PAYPAL_MODE", "sandbox")

    # Stripe
    stripe_secret_key: str = os.getenv("STRIPE_SECRET_KEY", "")
    stripe_webhook_secret: str = os.getenv("STRIPE_WEBHOOK_SECRET", "")

    # Przelewy24
    p24_merchant_id: str = os.getenv("P24_MERCHANT_ID", "")
    p24_pos_id: str = os.getenv("P24_POS_ID", "")
    p24_crc: str = os.getenv("P24_CRC", "")

    # Database
    db_url: str = os.getenv("DATABASE_URL", "sqlite:///./intentforge.db")

    # Camera / Vision
    opencv_dnn_path: str = os.getenv("OPENCV_DNN_PATH", "./models")
    rtsp_timeout: int = int(os.getenv("RTSP_TIMEOUT", "10"))

    # Storage
    upload_path: str = os.getenv("UPLOAD_PATH", "./uploads")

    # API Keys
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    openweather_api_key: str = os.getenv("OPENWEATHER_API_KEY", "")


config = ServiceConfig()


# =============================================================================
# Email Service
# =============================================================================


class EmailService:
    """Email service using SMTP configuration from .env"""

    TEMPLATES = {
        "contact_confirmation": {
            "subject": "Dziękujemy za kontakt",
            "body": """
Witaj {name}!

Otrzymaliśmy Twoją wiadomość i odpowiemy najszybciej jak to możliwe.

Twoja wiadomość:
{message}

Pozdrawiamy,
Zespół
""",
        },
        "contact_notification": {
            "subject": "Nowa wiadomość z formularza kontaktowego",
            "body": """
Nowa wiadomość z formularza:

Imię: {name}
Email: {email}
Telefon: {phone}
Temat: {subject}

Wiadomość:
{message}
""",
        },
        "ebook_purchase": {
            "subject": "Twój e-book jest gotowy do pobrania!",
            "body": """
Witaj!

Dziękujemy za zakup "{product_name}"!

Możesz pobrać swój e-book klikając w poniższy link:
{download_url}

Link jest ważny przez 7 dni.

ID zamówienia: {order_id}

Pozdrawiamy!
""",
        },
        "camera_alert": {
            "subject": "⚠️ Alert z kamery: {type}",
            "body": """
Wykryto zdarzenie na kamerze!

Typ: {type}
Kamera: {camera}
Czas: {timestamp}
Pewność: {confidence}%

Zrzut ekranu w załączniku.
""",
        },
    }

    def __init__(self):
        self.config = config

    async def send(
        self,
        to: str,
        subject: str | None = None,
        body: str | None = None,
        template: str | None = None,
        data: dict[str, Any] | None = None,
        attachments: list[str] | None = None,
    ) -> dict[str, Any]:
        """Send email using SMTP"""

        import smtplib
        from email import encoders
        from email.mime.base import MIMEBase
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        # Use template if specified
        if template and template in self.TEMPLATES:
            tpl = self.TEMPLATES[template]
            subject = tpl["subject"].format(**(data or {}))
            body = tpl["body"].format(**(data or {}))

        # Create message
        msg = MIMEMultipart()
        msg["From"] = self.config.smtp_from
        msg["To"] = to
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # Add attachments
        if attachments:
            for filepath in attachments:
                if os.path.exists(filepath):
                    with open(filepath, "rb") as f:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(f.read())
                        encoders.encode_base64(part)
                        part.add_header(
                            "Content-Disposition",
                            f"attachment; filename={os.path.basename(filepath)}",
                        )
                        msg.attach(part)

        # Send
        try:
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                server.starttls()
                if self.config.smtp_user and self.config.smtp_password:
                    server.login(self.config.smtp_user, self.config.smtp_password)
                server.send_message(msg)

            logger.info(f"Email sent to {to}")
            return {"success": True, "recipient": to}

        except Exception as e:
            logger.error(f"Email error: {e}")
            return {"success": False, "error": str(e)}

    async def send_template(self, template: str, to: str, data: dict[str, Any]) -> dict[str, Any]:
        return await self.send(to=to, template=template, data=data)


# =============================================================================
# Payment Service
# =============================================================================


class PaymentService:
    """Payment processing with PayPal, Stripe, P24"""

    def __init__(self):
        self.config = config

    async def checkout(
        self,
        amount: float,
        currency: str,
        product: str,
        email: str,
        provider: str = "paypal",
        return_url: str | None = None,
        cancel_url: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create payment checkout"""

        if provider == "paypal":
            return await self._paypal_checkout(
                amount, currency, product, email, return_url, cancel_url, metadata
            )
        elif provider == "stripe":
            return await self._stripe_checkout(
                amount, currency, product, email, return_url, cancel_url, metadata
            )
        elif provider == "przelewy24":
            return await self._p24_checkout(
                amount, currency, product, email, return_url, cancel_url, metadata
            )
        else:
            raise ValueError(f"Unknown payment provider: {provider}")

    async def _paypal_checkout(
        self, amount, currency, product, email, return_url, cancel_url, metadata
    ) -> dict[str, Any]:
        """PayPal checkout"""
        import httpx

        # Get access token
        auth_url = f"https://api-m.{'sandbox.' if self.config.paypal_mode == 'sandbox' else ''}paypal.com/v1/oauth2/token"

        async with httpx.AsyncClient() as client:
            auth_response = await client.post(
                auth_url,
                auth=(self.config.paypal_client_id, self.config.paypal_secret),
                data={"grant_type": "client_credentials"},
            )
            access_token = auth_response.json()["access_token"]

            # Create order
            order_url = f"https://api-m.{'sandbox.' if self.config.paypal_mode == 'sandbox' else ''}paypal.com/v2/checkout/orders"

            order_data = {
                "intent": "CAPTURE",
                "purchase_units": [
                    {
                        "amount": {"currency_code": currency, "value": str(amount)},
                        "description": product,
                    }
                ],
                "application_context": {"return_url": return_url, "cancel_url": cancel_url},
            }

            response = await client.post(
                order_url, headers={"Authorization": f"Bearer {access_token}"}, json=order_data
            )

            order = response.json()

            # Find approval URL
            approval_url = next(
                (link["href"] for link in order.get("links", []) if link["rel"] == "approve"), None
            )

            # Store order in database
            await self._store_order(
                order["id"], email, amount, currency, product, "paypal", metadata
            )

            return {
                "success": True,
                "payment_id": order["id"],
                "redirect_url": approval_url,
                "provider": "paypal",
            }

    async def _stripe_checkout(
        self, amount, currency, product, email, return_url, cancel_url, metadata
    ) -> dict[str, Any]:
        """Stripe checkout"""
        import stripe

        stripe.api_key = self.config.stripe_secret_key

        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[
                {
                    "price_data": {
                        "currency": currency.lower(),
                        "product_data": {
                            "name": product,
                        },
                        "unit_amount": int(amount * 100),
                    },
                    "quantity": 1,
                }
            ],
            mode="payment",
            success_url=return_url + "?payment=success&payment_id={CHECKOUT_SESSION_ID}",
            cancel_url=cancel_url + "?payment=cancelled",
            customer_email=email,
            metadata=metadata or {},
        )

        await self._store_order(session.id, email, amount, currency, product, "stripe", metadata)

        return {
            "success": True,
            "payment_id": session.id,
            "redirect_url": session.url,
            "provider": "stripe",
        }

    async def _p24_checkout(
        self, amount, currency, product, email, return_url, cancel_url, metadata
    ) -> dict[str, Any]:
        """Przelewy24 checkout"""
        import hashlib

        import httpx

        session_id = f"p24_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(4).hex()}"

        # Calculate CRC
        crc_string = f"{session_id}|{self.config.p24_merchant_id}|{int(amount * 100)}|{currency}|{self.config.p24_crc}"
        sign = hashlib.sha384(crc_string.encode()).hexdigest()

        data = {
            "merchantId": int(self.config.p24_merchant_id),
            "posId": int(self.config.p24_pos_id),
            "sessionId": session_id,
            "amount": int(amount * 100),
            "currency": currency,
            "description": product,
            "email": email,
            "urlReturn": return_url,
            "urlStatus": f"{return_url.rsplit('/', 1)[0]}/api/webhook/p24",
            "sign": sign,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://sandbox.przelewy24.pl/api/v1/transaction/register",
                json=data,
                auth=(str(self.config.p24_pos_id), self.config.p24_crc),
            )

            result = response.json()

            if result.get("data", {}).get("token"):
                token = result["data"]["token"]
                await self._store_order(
                    session_id, email, amount, currency, product, "p24", metadata
                )

                return {
                    "success": True,
                    "payment_id": session_id,
                    "redirect_url": f"https://sandbox.przelewy24.pl/trnRequest/{token}",
                    "provider": "przelewy24",
                }

            return {"success": False, "error": result.get("error", "Unknown error")}

    async def verify(self, payment_id: str) -> dict[str, Any]:
        """Verify payment status"""
        # Check database for order status
        order = await self._get_order(payment_id)

        if not order:
            return {"success": False, "error": "Order not found"}

        return {
            "success": True,
            "status": order.get("status", "pending"),
            "email": order.get("email"),
            "order_id": payment_id,
            "download_url": order.get("download_url"),
        }

    async def _store_order(self, order_id, email, amount, currency, product, provider, metadata):
        """Store order in database"""
        # Implementation depends on your database
        pass

    async def _get_order(self, order_id) -> dict | None:
        """Get order from database"""
        # Implementation depends on your database
        return None


# =============================================================================
# Form Service
# =============================================================================


class FormService:
    """Form handling service"""

    def __init__(self):
        self.email_service = EmailService()

    async def submit(
        self, form_id: str, data: dict[str, Any], notify_email: str | None = None
    ) -> dict[str, Any]:
        """Handle form submission"""

        # Validate data
        validated = self._validate(form_id, data)
        if not validated["valid"]:
            return {"success": False, "errors": validated["errors"]}

        # Store in database
        record_id = await self._store(form_id, data)

        # Send notifications
        email_sent = False
        if notify_email or os.getenv("ADMIN_EMAIL"):
            admin_email = notify_email or os.getenv("ADMIN_EMAIL")
            await self.email_service.send(
                to=admin_email, template="contact_notification", data=data
            )

        # Send confirmation to user if email provided
        if data.get("email"):
            result = await self.email_service.send(
                to=data["email"], template="contact_confirmation", data=data
            )
            email_sent = result.get("success", False)

        return {
            "success": True,
            "record_id": record_id,
            "email_sent": email_sent,
            "recipient": data.get("email"),
        }

    def _validate(self, form_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Validate form data"""
        errors = []

        # Basic validation rules per form
        rules = {
            "contact": {"required": ["name", "email", "message"], "email_fields": ["email"]},
            "newsletter": {"required": ["email"], "email_fields": ["email"]},
        }

        form_rules = rules.get(form_id, {"required": [], "email_fields": []})

        # Check required fields
        for field in form_rules["required"]:
            if not data.get(field):
                errors.append(f"Pole '{field}' jest wymagane")

        # Validate email format
        import re

        email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

        for field in form_rules["email_fields"]:
            if data.get(field) and not email_pattern.match(data[field]):
                errors.append(f"Nieprawidłowy format email w polu '{field}'")

        return {"valid": len(errors) == 0, "errors": errors}

    async def _store(self, form_id: str, data: dict[str, Any]) -> str:
        """Store form submission in database"""
        # Implementation depends on your database
        record_id = f"{form_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        return record_id


# =============================================================================
# Camera Service
# =============================================================================


class CameraService:
    """Camera and image analysis service"""

    def __init__(self):
        self.config = config
        self.email_service = EmailService()
        self._detection_model = None

    async def analyze(self, source: str, detect: list[str] | None = None) -> dict[str, Any]:
        """Analyze camera frame"""

        detect = detect or ["motion", "objects"]

        # Capture frame
        frame = await self._capture_frame(source)
        if frame is None:
            return {"success": False, "error": "Could not capture frame"}

        detections = []

        # Motion detection
        if "motion" in detect:
            motion = await self._detect_motion(frame)
            if motion:
                detections.extend(motion)

        # Object detection
        if "objects" in detect or "person" in detect or "vehicle" in detect:
            objects = await self._detect_objects(frame, detect)
            if objects:
                detections.extend(objects)

        return {
            "success": True,
            "detections": detections,
            "timestamp": datetime.now().isoformat(),
            "frame_size": {"width": frame.shape[1], "height": frame.shape[0]},
        }

    async def snapshot(self, source: str, save: bool = False) -> dict[str, Any]:
        """Capture snapshot from camera"""
        import base64

        import cv2

        frame = await self._capture_frame(source)
        if frame is None:
            return {"success": False, "error": "Could not capture frame"}

        # Encode to JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        image_base64 = base64.b64encode(buffer).decode("utf-8")

        result = {"success": True, "image": image_base64, "timestamp": datetime.now().isoformat()}

        if save:
            filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(self.config.upload_path, filename)
            cv2.imwrite(filepath, frame)
            result["url"] = f"/uploads/{filename}"

        return result

    async def _capture_frame(self, source: str):
        """Capture frame from RTSP or file"""
        import cv2

        try:
            cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ret, frame = cap.read()
            cap.release()

            return frame if ret else None
        except Exception as e:
            logger.error(f"Capture error: {e}")
            return None

    async def _detect_motion(self, frame) -> list[dict]:
        """Simple motion detection using frame difference"""
        # This is a simplified version - in production use background subtraction
        return []

    async def _detect_objects(self, frame, detect_types: list[str]) -> list[dict]:
        """Object detection using pre-trained model"""
        import cv2

        detections = []

        # Load model if not loaded
        if self._detection_model is None:
            model_path = os.path.join(self.config.opencv_dnn_path, "yolov4-tiny.weights")
            config_path = os.path.join(self.config.opencv_dnn_path, "yolov4-tiny.cfg")

            if os.path.exists(model_path) and os.path.exists(config_path):
                self._detection_model = cv2.dnn.readNetFromDarknet(config_path, model_path)
            else:
                return []

        # Run detection
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self._detection_model.setInput(blob)

        layer_names = self._detection_model.getLayerNames()
        output_layers = [
            layer_names[i - 1] for i in self._detection_model.getUnconnectedOutLayers()
        ]
        outputs = self._detection_model.forward(output_layers)

        # Process detections
        height, width = frame.shape[:2]

        # COCO class names
        class_names = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
        ]

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(scores.argmax())
                confidence = float(scores[class_id])

                if confidence > 0.5:
                    class_name = class_names[class_id] if class_id < len(class_names) else "object"

                    # Filter by requested types
                    if "person" in detect_types and class_name != "person":
                        if "vehicle" not in detect_types or class_name not in [
                            "car",
                            "truck",
                            "bus",
                            "motorcycle",
                        ]:
                            continue

                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    detections.append(
                        {
                            "type": class_name,
                            "confidence": confidence,
                            "bbox": {
                                "x": (center_x - w / 2) / width,
                                "y": (center_y - h / 2) / height,
                                "width": w / width,
                                "height": h / height,
                            },
                        }
                    )

        return detections


# =============================================================================
# Chat Service (LLM)
# =============================================================================


class ChatService:
    """Chat service using LLM from .env configuration"""

    def __init__(self):
        self.config = config
        self._provider = None

    def _get_provider(self, model: str | None = None):
        """Get LLM provider based on .env configuration"""
        from .llm.providers import get_llm_provider

        provider_name = os.getenv("LLM_PROVIDER", "ollama")
        default_model = os.getenv("LLM_MODEL", "llama3.1:8b")

        return get_llm_provider(
            provider=provider_name,
            model=model or default_model,
        )

    def _get_cache_key(self, message: str, model: str, system: str) -> str:
        """Generate cache key for LLM response"""
        import hashlib

        key_data = f"{message}::{model}::{system}"
        return f"llm:{hashlib.sha256(key_data.encode()).hexdigest()[:32]}"

    def _get_cached_response(self, cache_key: str) -> dict | None:
        """Get cached LLM response from Redis"""
        try:
            import redis

            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            client = redis.from_url(redis_url)
            data = client.get(f"intentforge:{cache_key}")
            if data:
                import json

                return json.loads(data)
        except Exception as e:
            logger.debug(f"Cache miss or error: {e}")
        return None

    def _set_cached_response(self, cache_key: str, response: dict, ttl: int = 3600) -> None:
        """Cache LLM response in Redis"""
        try:
            import redis

            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            client = redis.from_url(redis_url)
            import json

            client.setex(f"intentforge:{cache_key}", ttl, json.dumps(response))
        except Exception as e:
            logger.debug(f"Cache set error: {e}")

    async def send(
        self,
        message: str,
        model: str | None = None,
        history: list[dict] | None = None,
        system: str | None = None,
        stream: bool = False,
        use_cache: bool = True,
        cache_ttl: int = 3600,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Send message to LLM and get response.

        Args:
            message: User message
            model: Optional model override (uses LLM_MODEL from .env if not specified)
            history: Conversation history [{"role": "user/assistant", "content": "..."}]
            system: System prompt
            stream: Whether to stream response (not implemented in this version)
            use_cache: Whether to use Redis cache for responses (default: True)
            cache_ttl: Cache TTL in seconds (default: 3600)

        Returns:
            dict with response, model, tokens, etc.
        """
        try:
            provider = self._get_provider(model)
            used_model = model or os.getenv("LLM_MODEL", "llama3.1:8b")

            # Build prompt with history
            prompt = message
            if history:
                # Format history into prompt
                history_text = "\n".join(
                    f"{'User' if h['role'] == 'user' else 'Assistant'}: {h['content']}"
                    for h in history[-10:]  # Last 10 messages
                )
                prompt = f"Previous conversation:\n{history_text}\n\nUser: {message}"

            # Default system prompt if not provided
            if not system:
                system = (
                    "Jesteś pomocnym asystentem AI. Odpowiadaj po polsku. "
                    "Gdy podajesz kod, używaj bloków kodu z odpowiednim językiem (```python, ```javascript, etc.)."
                )

            # Check cache (only for messages without history for consistency)
            cache_key = None
            if use_cache and not history:
                cache_key = self._get_cache_key(message, used_model, system)
                cached = self._get_cached_response(cache_key)
                if cached:
                    cached["cached"] = True
                    return cached

            response = await provider.generate(prompt, system=system, **kwargs)

            result = {
                "success": True,
                "response": response.content,
                "model": response.model,
                "provider": response.provider,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "total_tokens": response.total_tokens,
                "latency_ms": response.latency_ms,
                "cached": False,
            }

            # Cache the response (only for messages without history)
            if use_cache and cache_key and not history:
                self._set_cached_response(cache_key, result, cache_ttl)

            return result

        except Exception as e:
            logger.error(f"Chat error: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": f"Przepraszam, wystąpił błąd: {e!s}",
            }

    async def models(self) -> dict[str, Any]:
        """List available models (for Ollama)"""
        try:
            provider_name = os.getenv("LLM_PROVIDER", "ollama")

            default_model = os.getenv("LLM_MODEL", "llama3.1:8b")

            if provider_name == "ollama":
                from .llm.providers import LLMConfig, OllamaProvider

                config = LLMConfig.from_env()
                provider = OllamaProvider(config)
                models = await provider.list_models()
                return {
                    "success": True,
                    "models": models,
                    "provider": "ollama",
                    "default_model": default_model,
                }

            # For other providers, return configured model
            return {
                "success": True,
                "models": [default_model],
                "provider": provider_name,
                "default_model": default_model,
            }

        except Exception as e:
            logger.error(f"Models list error: {e}")
            return {"success": False, "error": str(e), "models": []}


# =============================================================================
# Analytics Service (LLM-powered)
# =============================================================================


class AnalyticsService:
    """Analytics service using LLM for data generation and NLP queries"""

    def __init__(self):
        self.config = config
        self.chat_service = None

    def _get_chat_service(self):
        if self.chat_service is None:
            self.chat_service = ChatService()
        return self.chat_service

    async def stats(self, period: str = "current_month", **kwargs) -> dict[str, Any]:
        """Get analytics stats - uses LLM to generate realistic data"""
        try:
            chat = self._get_chat_service()
            response = await chat.send(
                message=f"Generate realistic e-commerce analytics stats for {period}. "
                "Return JSON with: revenue (number), revenue_change (percent), orders (number), "
                "orders_change (percent), users (number), users_change (percent), "
                "conversion (percent), conversion_change (percent). Only return valid JSON.",
                system="You are a data generator. Return only valid JSON without markdown code blocks.",
            )

            if response.get("success"):
                import json

                try:
                    # Try to parse JSON from response
                    content = response["response"]
                    # Remove markdown if present
                    if "```" in content:
                        content = content.split("```")[1]
                        if content.startswith("json"):
                            content = content[4:]
                    return {"success": True, **json.loads(content.strip())}
                except json.JSONDecodeError:
                    pass

            # Fallback to generated data
            import random

            return {
                "success": True,
                "revenue": random.randint(80000, 150000),
                "revenue_change": round(random.uniform(-5, 20), 1),
                "orders": random.randint(1000, 3000),
                "orders_change": round(random.uniform(-10, 25), 1),
                "users": random.randint(3000, 8000),
                "users_change": round(random.uniform(-5, 30), 1),
                "conversion": round(random.uniform(2, 5), 1),
                "conversion_change": round(random.uniform(-2, 2), 1),
            }
        except Exception as e:
            logger.error(f"Analytics stats error: {e}")
            return {"success": False, "error": str(e)}

    async def chart_data(
        self, metric: str = "revenue", period: str = "week", **kwargs
    ) -> dict[str, Any]:
        """Get chart data for analytics"""
        import random

        periods_config = {
            "week": (["Pon", "Wt", "Śr", "Czw", "Pt", "Sob", "Ndz"], 7),
            "month": (["Tydz 1", "Tydz 2", "Tydz 3", "Tydz 4"], 4),
            "year": (
                [
                    "Sty",
                    "Lut",
                    "Mar",
                    "Kwi",
                    "Maj",
                    "Cze",
                    "Lip",
                    "Sie",
                    "Wrz",
                    "Paź",
                    "Lis",
                    "Gru",
                ],
                12,
            ),
        }

        labels, count = periods_config.get(period, periods_config["week"])
        base = 10000 if metric == "revenue" else 100

        return {
            "success": True,
            "labels": labels,
            "data": [random.randint(int(base * 0.7), int(base * 1.5)) for _ in range(count)],
        }

    async def activities(self, limit: int = 10, **kwargs) -> dict[str, Any]:
        """Get recent activities"""
        import random

        activity_types = [
            {"type": "order", "message": "Nowe zamówienie #{}", "icon": "📦"},
            {"type": "user", "message": "Nowy użytkownik: user{}@example.com", "icon": "👤"},
            {"type": "payment", "message": "Płatność {} PLN potwierdzona", "icon": "💳"},
            {"type": "review", "message": "Nowa recenzja produktu ({}⭐)", "icon": "⭐"},
            {"type": "alert", "message": "Niski stan magazynowy: Produkt {}", "icon": "⚠️"},
        ]

        activities = []
        for _i in range(limit):
            activity = random.choice(activity_types)
            time_ago = random.randint(1, 60)
            activities.append(
                {
                    "type": activity["type"],
                    "message": activity["message"].format(random.randint(100, 9999)),
                    "icon": activity["icon"],
                    "time": f"{time_ago} min temu",
                }
            )

        return {"success": True, "activities": activities}

    async def query(self, query: str, **kwargs) -> dict[str, Any]:
        """Process natural language analytics query using LLM"""
        try:
            chat = self._get_chat_service()
            response = await chat.send(
                message=f"Analyze this analytics query and provide insights: {query}",
                system="You are an analytics assistant. Provide helpful insights about business data. "
                "Be concise and actionable. Respond in Polish.",
            )

            return {
                "success": True,
                "message": response.get("response", "Nie udało się przetworzyć zapytania."),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def products(self, limit: int = 10, **kwargs) -> dict[str, Any]:
        """Get top products data"""
        import random

        products = [
            ('Laptop Pro 15"', "Elektronika"),
            ("Słuchawki BT", "Audio"),
            ("Smartwatch X", "Wearables"),
            ("Klawiatura RGB", "Akcesoria"),
            ('Monitor 27"', "Elektronika"),
            ("Mysz Gaming", "Akcesoria"),
            ('Tablet 10"', "Elektronika"),
            ("Głośnik BT", "Audio"),
            ("Kamera IP", "Smart Home"),
            ("Router WiFi 6", "Sieć"),
        ]

        result = []
        for i, (name, category) in enumerate(products[:limit], 1):
            sold = random.randint(50, 600)
            price = random.randint(100, 2000)
            result.append(
                {
                    "rank": i,
                    "name": name,
                    "category": category,
                    "sold": sold,
                    "revenue": f"{sold * price:,} PLN".replace(",", " "),
                }
            )

        return {"success": True, "products": result}


# =============================================================================
# Voice/NLP Service (LLM-powered)
# =============================================================================


class VoiceService:
    """Voice command processing using LLM for NLP"""

    def __init__(self):
        self.chat_service = None

    def _get_chat_service(self):
        if self.chat_service is None:
            self.chat_service = ChatService()
        return self.chat_service

    async def process(self, command: str, language: str = "pl", **kwargs) -> dict[str, Any]:
        """Process voice/text command using LLM to understand intent"""
        try:
            chat = self._get_chat_service()
            response = await chat.send(
                message=f'Parse this smart home voice command: "{command}"\n\n'
                "Return JSON with:\n"
                "- actions: array of {type: 'device'|'scene', device_id?: string, device_type?: string, scene?: string, state?: object}\n"
                "- response: human-readable response in Polish\n\n"
                "Device IDs: living_light, living_lamp, living_tv, living_ac, bed_light, bed_blinds, bed_fan, bed_speaker, kitchen_light, kitchen_coffee\n"
                "Scenes: morning, work, movie, night\n"
                "Only return valid JSON.",
                system="You are a smart home voice command parser. Parse commands and return structured JSON actions. "
                "Return only valid JSON without markdown code blocks.",
            )

            if response.get("success"):
                import json

                try:
                    content = response["response"]
                    if "```" in content:
                        content = content.split("```")[1]
                        if content.startswith("json"):
                            content = content[4:]
                    parsed = json.loads(content.strip())
                    return {"success": True, **parsed}
                except json.JSONDecodeError:
                    pass

            # Fallback - simple keyword parsing
            return await self._fallback_parse(command)

        except Exception as e:
            logger.error(f"Voice processing error: {e}")
            return await self._fallback_parse(command)

    async def _fallback_parse(self, command: str) -> dict[str, Any]:
        """Fallback command parsing without LLM"""
        cmd = command.lower()
        actions = []
        response = "Nie rozpoznano komendy."

        if "włącz" in cmd and "światło" in cmd:
            if "salon" in cmd:
                actions.append(
                    {"type": "device", "device_id": "living_light", "device_type": "light"}
                )
                response = "Włączam światło w salonie"
            elif "wszystk" in cmd:
                for device in ["living_light", "living_lamp", "bed_light", "kitchen_light"]:
                    actions.append({"type": "device", "device_id": device, "device_type": "light"})
                response = "Włączam wszystkie światła"

        elif "wyłącz" in cmd and "wszystko" in cmd:
            response = "Wyłączam wszystkie urządzenia"

        elif "scen" in cmd or "tryb" in cmd:
            if "film" in cmd:
                actions.append({"type": "scene", "scene": "movie"})
                response = "Aktywuję scenę filmową"
            elif "noc" in cmd:
                actions.append({"type": "scene", "scene": "night"})
                response = "Aktywuję tryb nocny"

        return {"success": True, "actions": actions, "response": response}


# =============================================================================
# File Processing Service (LLM-powered with Vision)
# =============================================================================


class FileService:
    """File processing service using LLM for content analysis and Vision for images"""

    def __init__(self):
        self.config = config
        self.chat_service = None
        # Vision settings from .env
        self.vision_provider = os.getenv("VISION_PROVIDER", "ollama")
        self.vision_model = os.getenv("VISION_MODEL", "llava:13b")
        self.vision_max_tokens = int(os.getenv("VISION_MAX_TOKENS", "2048"))
        self.vision_temperature = float(os.getenv("VISION_TEMPERATURE", "0.1"))
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def _get_chat_service(self):
        if self.chat_service is None:
            self.chat_service = ChatService()
        return self.chat_service

    async def _analyze_image_with_vision(
        self, image_base64: str, prompt: str | None = None
    ) -> dict[str, Any]:
        """Analyze image using Ollama Vision (LLaVA)"""
        import httpx

        prompt = prompt or (
            "Przeanalizuj ten obraz. Opisz co widzisz, wykryj obiekty, "
            "rozpoznaj tekst (OCR) jeśli jest widoczny. Odpowiedz po polsku."
        )

        payload = {
            "model": self.vision_model,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
            "options": {
                "temperature": self.vision_temperature,
                "num_predict": self.vision_max_tokens,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.ollama_host}/api/generate",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                return {
                    "success": True,
                    "response": data.get("response", ""),
                    "model": self.vision_model,
                    "provider": "ollama-vision",
                }

        except httpx.TimeoutException:
            return {"success": False, "error": "Vision API timeout - model may be loading"}
        except Exception as e:
            logger.error(f"Vision API error: {e}")
            return {"success": False, "error": str(e)}

    async def analyze(
        self,
        filename: str,
        content: str | None = None,
        image_base64: str | None = None,
        file_type: str | None = None,
        options: dict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Analyze file content using LLM or Vision for images"""
        options = options or {}

        try:
            # Image analysis with Vision (LLaVA)
            if image_base64 or (file_type and file_type.startswith("image/")):
                if image_base64:
                    vision_result = await self._analyze_image_with_vision(
                        image_base64,
                        prompt="Przeanalizuj ten obraz szczegółowo. "
                        "1. Opisz co widzisz na obrazie. "
                        "2. Wykryj i wymień wszystkie obiekty. "
                        "3. Rozpoznaj i przepisz cały widoczny tekst (OCR). "
                        "4. Określ kolory, styl i nastrój obrazu. "
                        "Odpowiedz po polsku w formacie strukturalnym.",
                    )

                    if vision_result.get("success"):
                        response_text = vision_result.get("response", "")
                        return {
                            "success": True,
                            "filename": filename,
                            "analysis": {
                                "description": response_text,
                                "model": self.vision_model,
                                "provider": "ollama-vision",
                            },
                        }
                    else:
                        return {
                            "success": False,
                            "filename": filename,
                            "error": vision_result.get("error", "Vision analysis failed"),
                        }
                else:
                    return {
                        "success": False,
                        "filename": filename,
                        "error": "Brak danych obrazu (image_base64). Wyślij obraz zakodowany w base64.",
                    }

            # Text content analysis with LLM
            chat = self._get_chat_service()

            if options.get("analyze") and content:
                response = await chat.send(
                    message=f"Analyze this {file_type or 'file'} content and provide insights:\n\n{content[:2000]}",
                    system="You are a file content analyzer. Provide structured analysis including: "
                    "description, key findings, data summary. Respond in Polish.",
                )

                return {
                    "success": True,
                    "filename": filename,
                    "analysis": {
                        "description": response.get("response", "Brak analizy"),
                    },
                }

            return {"success": True, "filename": filename, "message": "File received"}

        except Exception as e:
            logger.error(f"File analysis error: {e}")
            return {"success": False, "error": str(e)}

    async def ocr(
        self, image_base64: str | None = None, use_tesseract: bool = True, **kwargs
    ) -> dict[str, Any]:
        """
        Two-phase OCR processing:
        1. Tesseract for precise text extraction
        2. Vision (LLaVA) as fallback or for context
        """
        if not image_base64:
            return {"success": False, "error": "Brak danych obrazu (image_base64)"}

        ocr_text = ""
        ocr_method = "vision"

        # Phase 1: Try Tesseract OCR first (more precise for documents)
        if use_tesseract:
            tesseract_result = await self._ocr_with_tesseract(image_base64)
            if tesseract_result.get("success") and tesseract_result.get("text", "").strip():
                ocr_text = tesseract_result["text"]
                ocr_method = "tesseract"

        # Phase 2: Use Vision as fallback or supplement
        if not ocr_text.strip():
            result = await self._analyze_image_with_vision(
                image_base64,
                prompt="Rozpoznaj i przepisz CAŁY tekst widoczny na tym obrazie. "
                "Zachowaj oryginalny układ tekstu. "
                "Jeśli nie ma tekstu, napisz 'Brak tekstu na obrazie'. "
                "Odpowiedz tylko tekstem z obrazu, bez dodatkowych komentarzy.",
            )
            if result.get("success"):
                ocr_text = result.get("response", "")
                ocr_method = "vision"

        if ocr_text.strip():
            return {
                "success": True,
                "text": ocr_text,
                "method": ocr_method,
                "model": self.vision_model if ocr_method == "vision" else "tesseract",
            }
        else:
            return {"success": False, "error": "OCR failed - no text detected"}

    def _preprocess_image_for_ocr(self, image):
        """Preprocess image for better OCR results"""
        from PIL import ImageEnhance, ImageFilter

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert to grayscale
        gray = image.convert("L")

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(2.0)

        # Sharpen
        gray = gray.filter(ImageFilter.SHARPEN)

        # Binarize (threshold)
        threshold = 128
        gray = gray.point(lambda x: 255 if x > threshold else 0, "1")

        return gray

    async def _ocr_with_tesseract(
        self, image_base64: str, with_confidence: bool = True
    ) -> dict[str, Any]:
        """OCR using Tesseract (pytesseract) with confidence scores and preprocessing"""
        try:
            import base64
            import io

            from PIL import Image

            # Decode base64 to image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))

            # Preprocess image for better OCR
            processed_image = self._preprocess_image_for_ocr(image)

            # Try to import pytesseract
            try:
                import pytesseract

                # Try multiple PSM modes for best results
                psm_modes = [6, 3, 4, 11]  # 6=block, 3=auto, 4=single column, 11=sparse
                best_result = {"text": "", "confidence": 0}

                for psm in psm_modes:
                    custom_config = f"--oem 3 --psm {psm} -l pol+eng"

                    try:
                        # Try with preprocessed image first
                        data = pytesseract.image_to_data(
                            processed_image,
                            config=custom_config,
                            output_type=pytesseract.Output.DICT,
                        )

                        words = []
                        confidences = []

                        for i, conf in enumerate(data["conf"]):
                            if conf != -1:
                                word = data["text"][i].strip()
                                if word:
                                    words.append(word)
                                    confidences.append(float(conf))

                        text = " ".join(words)
                        avg_conf = sum(confidences) / len(confidences) if confidences else 0

                        if (
                            len(text) > len(best_result["text"])
                            or avg_conf > best_result["confidence"]
                        ):
                            best_result = {"text": text, "confidence": avg_conf, "psm": psm}

                        # If we got good results, stop trying
                        if len(text) > 50 and avg_conf > 70:
                            break

                    except Exception:
                        continue

                # If preprocessed didn't work well, try original image
                if len(best_result["text"]) < 20:
                    for psm in [6, 3]:
                        custom_config = f"--oem 3 --psm {psm} -l pol+eng"
                        try:
                            data = pytesseract.image_to_data(
                                image, config=custom_config, output_type=pytesseract.Output.DICT
                            )

                            words = []
                            confidences = []

                            for i, conf in enumerate(data["conf"]):
                                if conf != -1:
                                    word = data["text"][i].strip()
                                    if word:
                                        words.append(word)
                                        confidences.append(float(conf))

                            text = " ".join(words)
                            avg_conf = sum(confidences) / len(confidences) if confidences else 0

                            if len(text) > len(best_result["text"]):
                                best_result = {
                                    "text": text,
                                    "confidence": avg_conf,
                                    "psm": psm,
                                    "preprocessed": False,
                                }
                        except Exception:
                            continue

                result = {"success": True}
                result["text"] = best_result.get("text", "")
                result["confidence"] = round(best_result.get("confidence", 0), 2)
                result["word_count"] = len(best_result.get("text", "").split())

                if with_confidence:
                    # Recalculate with best config
                    custom_config = f"--oem 3 --psm {best_result.get('psm', 6)} -l pol+eng"
                    data = pytesseract.image_to_data(
                        processed_image, config=custom_config, output_type=pytesseract.Output.DICT
                    )

                    words = []
                    confidences = []

                    for i, conf in enumerate(data["conf"]):
                        if conf != -1:
                            word = data["text"][i].strip()
                            if word:
                                words.append(word)
                                confidences.append(float(conf))

                    text = " ".join(words)
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

                    result["text"] = text
                    result["confidence"] = round(avg_confidence, 2)
                    result["word_count"] = len(words)
                    result["low_confidence_words"] = sum(1 for c in confidences if c < 60)
                else:
                    text = pytesseract.image_to_string(image, config=custom_config)
                    result["text"] = text.strip()

                return result

            except ImportError:
                logger.warning("pytesseract not installed, falling back to Vision OCR")
                return {"success": False, "error": "pytesseract not installed"}

        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
            return {"success": False, "error": str(e)}

    async def process_document(self, image_base64: str | None = None, **kwargs) -> dict[str, Any]:
        """
        Full document processing pipeline:
        1. Vision analysis - detect document type (receipt, invoice, ID, etc.)
        2. Tesseract OCR - precise text extraction
        3. LLM - structure extracted data based on document type
        """
        if not image_base64:
            return {"success": False, "error": "Brak danych obrazu (image_base64)"}

        result = {
            "success": True,
            "phases": {},
        }

        # Phase 1: Vision - detect document type and initial analysis
        logger.info("Document processing: Phase 1 - Vision analysis")
        vision_result = await self._analyze_image_with_vision(
            image_base64,
            prompt="Przeanalizuj ten obraz dokumentu. Określ:\n"
            "1. TYP DOKUMENTU: (paragon, faktura, rachunek, dowód osobisty, prawo jazdy, "
            "umowa, list, formularz, inny)\n"
            "2. JĘZYK: główny język dokumentu\n"
            "3. JAKOŚĆ: (dobra, średnia, słaba)\n"
            "4. ORIENTACJA: (pionowa, pozioma, obrócona)\n"
            "5. KRÓTKI OPIS: co zawiera dokument\n"
            "Odpowiedz w formacie strukturalnym po polsku.",
        )

        doc_type = "unknown"
        if vision_result.get("success"):
            vision_text = vision_result.get("response", "").lower()
            result["phases"]["vision"] = {
                "success": True,
                "analysis": vision_result.get("response", ""),
                "model": self.vision_model,
            }

            # Detect document type from vision response
            if any(w in vision_text for w in ["paragon", "receipt", "kasowy"]):
                doc_type = "receipt"
            elif any(w in vision_text for w in ["faktura", "invoice", "vat"]):
                doc_type = "invoice"
            elif any(w in vision_text for w in ["dowód osobisty", "id card", "pesel"]):
                doc_type = "id_card"
            elif any(w in vision_text for w in ["prawo jazdy", "driver", "license"]):
                doc_type = "drivers_license"
            elif any(w in vision_text for w in ["rachunek", "bill", "opłata"]):
                doc_type = "bill"
            elif any(w in vision_text for w in ["umowa", "contract", "agreement"]):
                doc_type = "contract"
            else:
                doc_type = "document"

            result["document_type"] = doc_type
        else:
            result["phases"]["vision"] = {"success": False, "error": vision_result.get("error")}

        # Phase 2: Tesseract OCR - precise text extraction
        logger.info("Document processing: Phase 2 - Tesseract OCR")
        ocr_result = await self.ocr(image_base64, use_tesseract=True)

        ocr_text = ""
        if ocr_result.get("success"):
            ocr_text = ocr_result.get("text", "")
            result["phases"]["ocr"] = {
                "success": True,
                "text": ocr_text,
                "method": ocr_result.get("method", "unknown"),
                "char_count": len(ocr_text),
            }
        else:
            result["phases"]["ocr"] = {"success": False, "error": ocr_result.get("error")}

        # Phase 3: LLM - structure extracted data based on document type
        if ocr_text.strip():
            logger.info(f"Document processing: Phase 3 - LLM extraction for {doc_type}")
            extraction_result = await self._extract_document_data(ocr_text, doc_type)

            if extraction_result.get("success"):
                result["phases"]["extraction"] = {
                    "success": True,
                    "structured_data": extraction_result.get("data", {}),
                }
                result["extracted_data"] = extraction_result.get("data", {})
            else:
                result["phases"]["extraction"] = {
                    "success": False,
                    "error": extraction_result.get("error"),
                }

        result["raw_text"] = ocr_text
        return result

    async def _extract_document_data(self, ocr_text: str, doc_type: str) -> dict[str, Any]:
        """Extract structured data from OCR text based on document type"""
        chat = self._get_chat_service()

        # Document-specific extraction prompts
        prompts = {
            "receipt": (
                "Wyodrębnij dane z tego paragonu/rachunku:\n"
                "- nazwa_sklepu: nazwa sprzedawcy\n"
                "- adres: adres sklepu\n"
                "- nip: NIP sprzedawcy\n"
                "- data: data zakupu\n"
                "- godzina: godzina zakupu\n"
                "- produkty: lista [{nazwa, ilość, cena_jednostkowa, wartość}]\n"
                "- suma: kwota całkowita\n"
                "- platnosc: metoda płatności\n"
                "- numer_paragonu: numer dokumentu\n"
            ),
            "invoice": (
                "Wyodrębnij dane z tej faktury:\n"
                "- numer_faktury: numer dokumentu\n"
                "- data_wystawienia: data wystawienia\n"
                "- data_sprzedazy: data sprzedaży\n"
                "- termin_platnosci: termin płatności\n"
                "- sprzedawca: {nazwa, adres, nip, regon}\n"
                "- nabywca: {nazwa, adres, nip}\n"
                "- pozycje: lista [{nazwa, jm, ilosc, cena_netto, vat_stawka, wartosc_netto, wartosc_brutto}]\n"
                "- suma_netto: suma netto\n"
                "- suma_vat: suma VAT\n"
                "- suma_brutto: suma brutto\n"
                "- numer_konta: numer konta bankowego\n"
            ),
            "id_card": (
                "Wyodrębnij dane z tego dowodu osobistego:\n"
                "- imiona: imiona\n"
                "- nazwisko: nazwisko\n"
                "- data_urodzenia: data urodzenia\n"
                "- pesel: numer PESEL\n"
                "- plec: płeć\n"
                "- obywatelstwo: obywatelstwo\n"
                "- numer_dokumentu: numer dowodu\n"
                "- data_waznosci: data ważności\n"
                "- organ_wydajacy: organ wydający\n"
            ),
            "drivers_license": (
                "Wyodrębnij dane z tego prawa jazdy:\n"
                "- imiona: imiona\n"
                "- nazwisko: nazwisko\n"
                "- data_urodzenia: data urodzenia\n"
                "- miejsce_urodzenia: miejsce urodzenia\n"
                "- numer_dokumentu: numer prawa jazdy\n"
                "- kategorie: kategorie uprawnień\n"
                "- data_wydania: data wydania\n"
                "- data_waznosci: data ważności\n"
                "- organ_wydajacy: organ wydający\n"
            ),
            "bill": (
                "Wyodrębnij dane z tego rachunku:\n"
                "- wystawca: nazwa firmy wystawiającej\n"
                "- numer_klienta: numer klienta/abonenta\n"
                "- okres_rozliczeniowy: okres rozliczeniowy\n"
                "- kwota_do_zaplaty: kwota do zapłaty\n"
                "- termin_platnosci: termin płatności\n"
                "- numer_konta: numer konta do wpłaty\n"
                "- szczegoly: szczegóły rozliczenia\n"
            ),
            "document": (
                "Wyodrębnij kluczowe dane z tego dokumentu:\n"
                "- typ: typ dokumentu\n"
                "- tytul: tytuł/nagłówek\n"
                "- data: data dokumentu\n"
                "- nadawca: nadawca/wystawca\n"
                "- odbiorca: odbiorca\n"
                "- tresc_glowna: główna treść/podsumowanie\n"
                "- kwoty: wymienione kwoty\n"
                "- daty: wymienione daty\n"
            ),
        }

        prompt = prompts.get(doc_type, prompts["document"])

        try:
            response = await chat.send(
                message=f"{prompt}\n\nTekst dokumentu:\n{ocr_text[:3000]}",
                system="Jesteś ekspertem od ekstrakcji danych z dokumentów. "
                "Wyodrębnij dane i zwróć je jako poprawny JSON. "
                "Jeśli jakieś pole nie jest dostępne, użyj null. "
                "Zwróć TYLKO JSON, bez dodatkowego tekstu.",
            )

            if response.get("success"):
                import json

                try:
                    resp_content = response["response"]
                    # Clean markdown code blocks
                    if "```" in resp_content:
                        resp_content = resp_content.split("```")[1]
                        if resp_content.startswith("json"):
                            resp_content = resp_content[4:]
                    data = json.loads(resp_content.strip())
                    return {"success": True, "data": data}
                except json.JSONDecodeError:
                    # Return raw response if JSON parsing fails
                    return {"success": True, "data": {"raw_extraction": response["response"]}}

            return {"success": False, "error": "LLM extraction failed"}

        except Exception as e:
            logger.error(f"Document extraction error: {e}")
            return {"success": False, "error": str(e)}

    async def describe(self, image_base64: str | None = None, **kwargs) -> dict[str, Any]:
        """Describe image content using Vision"""
        if not image_base64:
            return {"success": False, "error": "Brak danych obrazu (image_base64)"}

        result = await self._analyze_image_with_vision(
            image_base64,
            prompt="Opisz szczegółowo co widzisz na tym obrazie. "
            "Wymień wszystkie obiekty, osoby, kolory, tło. "
            "Odpowiedz po polsku.",
        )

        if result.get("success"):
            return {
                "success": True,
                "description": result.get("response", ""),
                "model": self.vision_model,
            }
        else:
            return {"success": False, "error": result.get("error", "Description failed")}

    async def detect_objects(self, image_base64: str | None = None, **kwargs) -> dict[str, Any]:
        """Detect objects in image using Vision"""
        if not image_base64:
            return {"success": False, "error": "Brak danych obrazu (image_base64)"}

        result = await self._analyze_image_with_vision(
            image_base64,
            prompt="Wykryj i wymień wszystkie obiekty widoczne na tym obrazie. "
            "Dla każdego obiektu podaj: nazwę, przybliżoną lokalizację (góra/dół/lewo/prawo/środek), "
            "i pewność wykrycia (wysoka/średnia/niska). "
            "Odpowiedz w formacie listy po polsku.",
        )

        if result.get("success"):
            return {
                "success": True,
                "objects": result.get("response", ""),
                "model": self.vision_model,
            }
        else:
            return {"success": False, "error": result.get("error", "Object detection failed")}

    async def extract(self, filename: str, content: str | None = None, **kwargs) -> dict[str, Any]:
        """Extract structured data from file using LLM"""
        try:
            if not content:
                return {"success": False, "error": "No content provided"}

            chat = self._get_chat_service()
            response = await chat.send(
                message=f"Extract structured data from this content. Return as JSON:\n\n{content[:2000]}",
                system="You are a data extraction assistant. Extract key-value pairs, tables, "
                "and structured information. Return valid JSON only.",
            )

            if response.get("success"):
                import json

                try:
                    resp_content = response["response"]
                    if "```" in resp_content:
                        resp_content = resp_content.split("```")[1]
                        if resp_content.startswith("json"):
                            resp_content = resp_content[4:]
                    return {"success": True, "extracted_data": json.loads(resp_content.strip())}
                except json.JSONDecodeError:
                    return {"success": True, "extracted_data": {"raw": response["response"]}}

            return {"success": False, "error": "Extraction failed"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def pdf_ocr(
        self,
        pdf_base64: str | None = None,
        pdf_path: str | None = None,
        pages: list[int] | None = None,
        dpi: int = 200,
        **kwargs,
    ) -> dict[str, Any]:
        """
        OCR for multi-page PDF documents.

        Args:
            pdf_base64: PDF file encoded as base64
            pdf_path: Path to PDF file (alternative to base64)
            pages: List of page numbers to process (1-indexed), None = all pages
            dpi: Resolution for PDF to image conversion (default: 200)

        Returns:
            dict with pages array containing OCR results for each page
        """
        try:
            import base64
            import io

            # Try to import pdf2image
            try:
                from pdf2image import convert_from_bytes, convert_from_path
            except ImportError:
                return {
                    "success": False,
                    "error": "pdf2image not installed. Run: pip install pdf2image",
                    "hint": "Also install poppler: apt-get install poppler-utils",
                }

            # Convert PDF to images
            if pdf_base64:
                pdf_bytes = base64.b64decode(pdf_base64)
                images = convert_from_bytes(pdf_bytes, dpi=dpi)
            elif pdf_path:
                images = convert_from_path(pdf_path, dpi=dpi)
            else:
                return {"success": False, "error": "No PDF provided (pdf_base64 or pdf_path)"}

            total_pages = len(images)

            # Filter pages if specified
            if pages:
                pages_to_process = [(i, img) for i, img in enumerate(images, 1) if i in pages]
            else:
                pages_to_process = list(enumerate(images, 1))

            results = {
                "success": True,
                "total_pages": total_pages,
                "processed_pages": len(pages_to_process),
                "pages": [],
                "full_text": "",
            }

            all_text = []

            for page_num, image in pages_to_process:
                # Convert PIL image to base64
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode()

                # Run OCR on this page
                ocr_result = await self.ocr(image_base64=img_base64, use_tesseract=True)

                page_result = {
                    "page": page_num,
                    "success": ocr_result.get("success", False),
                    "text": ocr_result.get("text", ""),
                    "method": ocr_result.get("method", "unknown"),
                    "char_count": len(ocr_result.get("text", "")),
                }

                if ocr_result.get("confidence"):
                    page_result["confidence"] = ocr_result["confidence"]

                results["pages"].append(page_result)

                if ocr_result.get("text"):
                    all_text.append(f"--- Page {page_num} ---\n{ocr_result['text']}")

            results["full_text"] = "\n\n".join(all_text)
            results["total_chars"] = len(results["full_text"])

            return results

        except Exception as e:
            logger.error(f"PDF OCR error: {e}")
            return {"success": False, "error": str(e)}

    async def pdf_process(
        self,
        pdf_base64: str | None = None,
        pdf_path: str | None = None,
        extract_data: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Full PDF document processing with OCR and data extraction.

        Args:
            pdf_base64: PDF file encoded as base64
            pdf_path: Path to PDF file
            extract_data: Whether to extract structured data using LLM

        Returns:
            dict with OCR results and extracted data
        """
        # First, run OCR on all pages
        ocr_result = await self.pdf_ocr(pdf_base64=pdf_base64, pdf_path=pdf_path, **kwargs)

        if not ocr_result.get("success"):
            return ocr_result

        result = {
            "success": True,
            "total_pages": ocr_result["total_pages"],
            "processed_pages": ocr_result["processed_pages"],
            "pages": ocr_result["pages"],
            "full_text": ocr_result["full_text"],
            "total_chars": ocr_result.get("total_chars", 0),
        }

        # Extract structured data if requested
        if extract_data and ocr_result.get("full_text"):
            # Detect document type from first page
            first_page_text = ocr_result["pages"][0]["text"] if ocr_result["pages"] else ""

            doc_type = "document"
            text_lower = first_page_text.lower()
            if any(w in text_lower for w in ["faktura", "invoice", "vat"]):
                doc_type = "invoice"
            elif any(w in text_lower for w in ["umowa", "contract", "agreement"]):
                doc_type = "contract"
            elif any(w in text_lower for w in ["rachunek", "bill"]):
                doc_type = "bill"

            result["document_type"] = doc_type

            # Extract data using LLM
            extraction_result = await self._extract_document_data(
                ocr_result["full_text"][:5000], doc_type
            )

            if extraction_result.get("success"):
                result["extracted_data"] = extraction_result.get("data", {})
            else:
                result["extraction_error"] = extraction_result.get("error")

        return result


# =============================================================================
# Data Service
# =============================================================================


class DataService:
    """Generic data operations service"""

    def __init__(self):
        self.config = config

    async def list(
        self,
        table: str,
        limit: int = 50,
        offset: int = 0,
        sort: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """List records from table"""
        # Implementation depends on your database
        return {"items": [], "total": 0}

    async def get(self, table: str, id: Any) -> dict[str, Any]:
        """Get single record"""
        return {}

    async def create(self, table: str, data: dict[str, Any]) -> dict[str, Any]:
        """Create new record"""
        return {"id": None}

    async def update(self, table: str, id: Any, data: dict[str, Any]) -> dict[str, Any]:
        """Update record"""
        return {"success": True}

    async def delete(self, table: str, id: Any) -> dict[str, Any]:
        """Delete record"""
        return {"success": True}

    async def query(
        self, table: str, sql: str | None = None, params: dict | None = None, **kwargs
    ) -> dict[str, Any]:
        """Execute custom query"""
        return {"data": None}


# =============================================================================
# Module Service (DSL integration for autonomous modules)
# =============================================================================


class ModuleService:
    """
    DSL Service for autonomous modules.

    Usage in DSL:
        module.list()
        module.create(name="parser", description="CSV parser")
        module.create_from_task(task="Create a JSON validator")
        module.build(name="parser")
        module.start(name="parser")
        module.execute(name="parser", data={...})
        module.stop(name="parser")
    """

    async def list(self) -> dict[str, Any]:
        """List all available modules"""
        from .modules import module_manager

        return {
            "success": True,
            "modules": module_manager.list_modules(),
        }

    async def create(
        self,
        name: str,
        description: str = "",
        code: str | None = None,
        requirements: list = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Create a new module"""
        from .modules import module_manager

        try:
            module_info = await module_manager.create_module(
                name=name,
                description=description,
                code=code,
                requirements=requirements or [],
            )
            return {
                "success": True,
                "module": module_info.name,
                "port": module_info.port,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def create_from_task(
        self, task: str, name: str | None = None, **kwargs
    ) -> dict[str, Any]:
        """Create module from natural language task description"""
        from .modules import module_manager

        try:
            module_info = await module_manager.create_from_llm(
                task_description=task,
                module_name=name,
            )
            return {
                "success": True,
                "module": module_info.name,
                "port": module_info.port,
                "description": module_info.description,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def build(self, name: str, **kwargs) -> dict[str, Any]:
        """Build module Docker image"""
        from .modules import module_manager

        try:
            success = await module_manager.build_module(name)
            return {"success": success, "module": name}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def start(self, name: str, **kwargs) -> dict[str, Any]:
        """Start module container"""
        from .modules import module_manager

        try:
            success = await module_manager.start_module(name)
            module = module_manager.modules.get(name)
            return {
                "success": success,
                "module": name,
                "port": module.port if module else None,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def stop(self, name: str, **kwargs) -> dict[str, Any]:
        """Stop module container"""
        from .modules import module_manager

        try:
            success = await module_manager.stop_module(name)
            return {"success": success, "module": name}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def execute(
        self, name: str, action: str = "execute", data: dict | None = None, **kwargs
    ) -> dict[str, Any]:
        """Execute module action"""
        from .modules import module_manager

        try:
            result = await module_manager.execute_module(
                name=name,
                action=action,
                data=data or {},
            )
            return {
                "success": result.success,
                "result": result.result,
                "error": result.error,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# =============================================================================
# Agent Service (Autonomous task execution)
# =============================================================================


class AgentService:
    """
    DSL Service for autonomous agent.

    Usage in DSL:
        agent.execute(task="Create a web scraper", context={...})
        agent.history()
    """

    async def execute(
        self, task: str, context: dict | None = None, max_steps: int = 10, **kwargs
    ) -> dict[str, Any]:
        """Execute a complex task autonomously"""
        from .autonomous import autonomous_agent

        try:
            result = await autonomous_agent.execute_task(
                task=task,
                context=context or {},
                max_steps=min(max_steps, 20),
            )
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def history(self, limit: int = 20, **kwargs) -> dict[str, Any]:
        """Get task execution history"""
        from .autonomous import autonomous_agent

        return {
            "success": True,
            "history": autonomous_agent.task_history[-limit:],
        }


# =============================================================================
# Service Registry
# =============================================================================


class ServiceRegistry:
    """Central registry for all services"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._services = {}
        return cls._instance

    def __init__(self):
        if not self._services:
            self._services = {
                "email": EmailService(),
                "payment": PaymentService(),
                "form": FormService(),
                "camera": CameraService(),
                "data": DataService(),
                "chat": ChatService(),
                "analytics": AnalyticsService(),
                "voice": VoiceService(),
                "file": FileService(),
                "module": ModuleService(),
                "agent": AgentService(),
            }

    def get(self, name: str):
        return self._services.get(name)

    def list(self) -> list[str]:
        """List all registered service names"""
        return list(self._services.keys())

    async def handle_request(
        self, service: str, action: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Route request to appropriate service"""
        svc = self.get(service)
        if not svc:
            return {"success": False, "error": f"Unknown service: {service}"}

        method = getattr(svc, action, None)
        if not method:
            return {"success": False, "error": f"Unknown action: {action}"}

        try:
            return await method(**data)
        except Exception as e:
            logger.error(f"Service error: {e}")
            return {"success": False, "error": str(e)}


# Global registry
services = ServiceRegistry()

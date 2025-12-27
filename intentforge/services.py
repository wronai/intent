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
        subject: str = None,
        body: str = None,
        template: str = None,
        data: dict[str, Any] = None,
        attachments: list[str] = None,
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
        return_url: str = None,
        cancel_url: str = None,
        metadata: dict[str, Any] = None,
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
        self, form_id: str, data: dict[str, Any], notify_email: str = None
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

    async def analyze(self, source: str, detect: list[str] = None) -> dict[str, Any]:
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
        sort: str = None,
        filters: dict[str, Any] = None,
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
        self, table: str, sql: str = None, params: dict = None, **kwargs
    ) -> dict[str, Any]:
        """Execute custom query"""
        return {"data": None}


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
            }

    def get(self, name: str):
        return self._services.get(name)

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

"""
MQTT Intent Broker - Universal communication layer
Enables any client (static HTML, CLI, mobile, IoT) to submit intents
"""

import json
import asyncio
import logging
from typing import Optional, Callable, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
import threading
import time

if TYPE_CHECKING:
    from .core import IntentForge, Intent

logger = logging.getLogger(__name__)


class MQTTTopic(Enum):
    """Standard MQTT topics for intent communication"""
    # Client -> Server
    INTENT_REQUEST = "intentforge/intent/request/{client_id}"
    
    # Server -> Client
    INTENT_RESPONSE = "intentforge/intent/response/{client_id}"
    INTENT_STATUS = "intentforge/intent/status/{client_id}"
    
    # Broadcast
    SYSTEM_STATUS = "intentforge/system/status"
    DISCOVERY = "intentforge/discovery"
    CAPABILITIES = "intentforge/capabilities"
    
    # Wildcard subscriptions
    ALL_REQUESTS = "intentforge/intent/request/+"
    ALL_EVENTS = "intentforge/events/#"


@dataclass 
class MQTTMessage:
    """MQTT message wrapper"""
    topic: str
    payload: Dict[str, Any]
    qos: int = 1
    retain: bool = False
    
    def to_json(self) -> bytes:
        return json.dumps(self.payload).encode('utf-8')
    
    @classmethod
    def from_bytes(cls, topic: str, data: bytes) -> "MQTTMessage":
        return cls(
            topic=topic,
            payload=json.loads(data.decode('utf-8'))
        )


class MQTTIntentBroker:
    """
    MQTT-based message broker for intent distribution
    Allows any client to submit intents via MQTT
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 1883,
        username: Optional[str] = None,
        password: Optional[str] = None,
        client_id: str = "intentforge-server",
        forge: Optional["IntentForge"] = None
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.client_id = client_id
        self.forge = forge
        
        self._client = None
        self._connected = False
        self._loop = None
        self._thread = None
        
        # Callbacks
        self._on_intent: Optional[Callable] = None
        self._message_handlers: Dict[str, Callable] = {}
    
    def connect(self) -> bool:
        """Connect to MQTT broker"""
        try:
            import paho.mqtt.client as mqtt
            
            self._client = mqtt.Client(
                client_id=self.client_id,
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2
            )
            
            if self.username and self.password:
                self._client.username_pw_set(self.username, self.password)
            
            # Set callbacks
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.on_message = self._on_message
            
            # Connect
            self._client.connect(self.host, self.port, keepalive=60)
            
            logger.info(f"Connecting to MQTT broker at {self.host}:{self.port}")
            return True
            
        except ImportError:
            logger.error("paho-mqtt not installed. Run: pip install paho-mqtt")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            return False
    
    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        """Handle connection"""
        if reason_code == 0:
            self._connected = True
            logger.info("Connected to MQTT broker")
            
            # Subscribe to intent requests
            client.subscribe(MQTTTopic.ALL_REQUESTS.value)
            client.subscribe(MQTTTopic.DISCOVERY.value)
            
            # Publish capabilities
            self._publish_capabilities()
        else:
            logger.error(f"Connection failed with code: {reason_code}")
    
    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties=None):
        """Handle disconnection"""
        self._connected = False
        logger.warning(f"Disconnected from MQTT broker: {reason_code}")
    
    def _on_message(self, client, userdata, msg):
        """Handle incoming messages"""
        try:
            message = MQTTMessage.from_bytes(msg.topic, msg.payload)
            
            # Route to appropriate handler
            if "/request/" in msg.topic:
                self._handle_intent_request(message, msg.topic)
            elif msg.topic == MQTTTopic.DISCOVERY.value:
                self._handle_discovery(message)
            else:
                # Check custom handlers
                for pattern, handler in self._message_handlers.items():
                    if self._topic_matches(msg.topic, pattern):
                        handler(message)
                        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in message: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    def _handle_intent_request(self, message: MQTTMessage, topic: str):
        """Process intent request from client"""
        # Extract client_id from topic
        parts = topic.split('/')
        client_id = parts[-1] if len(parts) > 0 else "unknown"
        
        logger.info(f"Received intent request from {client_id}")
        
        # Create intent from message
        from .core import Intent
        
        try:
            intent = Intent.from_dict(message.payload)
            
            # Process asynchronously
            if self.forge:
                asyncio.run_coroutine_threadsafe(
                    self._process_and_respond(intent, client_id),
                    self._loop
                )
            
        except Exception as e:
            logger.error(f"Failed to process intent: {e}")
            self._send_error(client_id, str(e))
    
    async def _process_and_respond(self, intent: "Intent", client_id: str):
        """Process intent and send response"""
        try:
            result = await self.forge.process_intent(intent)
            
            # Send response
            response_topic = MQTTTopic.INTENT_RESPONSE.value.format(client_id=client_id)
            self._publish(response_topic, result.to_dict())
            
        except Exception as e:
            self._send_error(client_id, str(e))
    
    def _send_error(self, client_id: str, error: str):
        """Send error response to client"""
        response_topic = MQTTTopic.INTENT_RESPONSE.value.format(client_id=client_id)
        self._publish(response_topic, {
            "success": False,
            "error": error
        })
    
    def _publish_capabilities(self):
        """Publish server capabilities"""
        from .core import IntentType, TargetPlatform
        
        capabilities = {
            "server_id": self.client_id,
            "version": "0.1.0",
            "intent_types": [t.value for t in IntentType],
            "target_platforms": [p.value for p in TargetPlatform],
            "features": [
                "code_generation",
                "validation",
                "caching",
                "auto_deploy"
            ]
        }
        
        self._publish(MQTTTopic.CAPABILITIES.value, capabilities, retain=True)
    
    def _handle_discovery(self, message: MQTTMessage):
        """Handle discovery request"""
        self._publish_capabilities()
    
    def _publish(self, topic: str, payload: Dict[str, Any], retain: bool = False):
        """Publish message to topic"""
        if self._client and self._connected:
            self._client.publish(
                topic,
                json.dumps(payload),
                qos=1,
                retain=retain
            )
    
    def _topic_matches(self, topic: str, pattern: str) -> bool:
        """Check if topic matches pattern (with wildcards)"""
        topic_parts = topic.split('/')
        pattern_parts = pattern.split('/')
        
        if len(pattern_parts) != len(topic_parts) and '#' not in pattern:
            return False
        
        for t, p in zip(topic_parts, pattern_parts):
            if p == '#':
                return True
            if p != '+' and p != t:
                return False
        
        return True
    
    def start_listening(self):
        """Start listening for messages in background thread"""
        if self._client:
            # Create event loop for async processing
            self._loop = asyncio.new_event_loop()
            
            # Start loop in separate thread
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            
            # Start MQTT loop
            self._client.loop_start()
            logger.info("MQTT listener started")
    
    def _run_loop(self):
        """Run async event loop in thread"""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
    
    def stop(self):
        """Stop the broker"""
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
        
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
    
    def disconnect(self):
        """Disconnect from broker"""
        self.stop()
    
    def register_handler(self, topic_pattern: str, handler: Callable):
        """Register custom message handler"""
        self._message_handlers[topic_pattern] = handler


# JavaScript client for static HTML pages
JS_CLIENT_CODE = '''
/**
 * IntentForge MQTT Client
 * Include this in static HTML pages to submit intents
 * 
 * Usage:
 *   const client = new IntentForgeClient('ws://localhost:9001');
 *   client.submitIntent({
 *     description: "Create endpoint to list users",
 *     intent_type: "api_endpoint"
 *   });
 */

class IntentForgeClient {
    constructor(brokerUrl, options = {}) {
        this.brokerUrl = brokerUrl;
        this.clientId = options.clientId || `client_${Date.now()}`;
        this.client = null;
        this.callbacks = new Map();
        this.connected = false;
        
        this.connect();
    }
    
    connect() {
        // Use MQTT.js for browser
        if (typeof mqtt === 'undefined') {
            console.error('MQTT.js not loaded. Add: <script src="https://unpkg.com/mqtt/dist/mqtt.min.js"></script>');
            return;
        }
        
        this.client = mqtt.connect(this.brokerUrl, {
            clientId: this.clientId,
            clean: true
        });
        
        this.client.on('connect', () => {
            this.connected = true;
            console.log('Connected to IntentForge');
            
            // Subscribe to responses
            this.client.subscribe(`intentforge/intent/response/${this.clientId}`);
        });
        
        this.client.on('message', (topic, payload) => {
            const message = JSON.parse(payload.toString());
            this._handleResponse(message);
        });
        
        this.client.on('error', (err) => {
            console.error('MQTT error:', err);
        });
    }
    
    submitIntent(intent) {
        return new Promise((resolve, reject) => {
            if (!this.connected) {
                reject(new Error('Not connected'));
                return;
            }
            
            const intentId = intent.intent_id || `intent_${Date.now()}`;
            intent.intent_id = intentId;
            
            // Store callback
            this.callbacks.set(intentId, { resolve, reject });
            
            // Publish intent
            const topic = `intentforge/intent/request/${this.clientId}`;
            this.client.publish(topic, JSON.stringify(intent));
            
            // Timeout
            setTimeout(() => {
                if (this.callbacks.has(intentId)) {
                    this.callbacks.delete(intentId);
                    reject(new Error('Intent timeout'));
                }
            }, 30000);
        });
    }
    
    _handleResponse(message) {
        const intentId = message.intent_id;
        const callback = this.callbacks.get(intentId);
        
        if (callback) {
            this.callbacks.delete(intentId);
            
            if (message.success) {
                callback.resolve(message);
            } else {
                callback.reject(new Error(message.error || 'Unknown error'));
            }
        }
    }
    
    disconnect() {
        if (this.client) {
            this.client.end();
        }
    }
}

// Make available globally
window.IntentForgeClient = IntentForgeClient;
'''


def get_js_client() -> str:
    """Get JavaScript client code for embedding in HTML"""
    return JS_CLIENT_CODE

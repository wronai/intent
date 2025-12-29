#!/usr/bin/env python3
"""
IntentForge MQTT Real-time Example
==================================

Demonstrates real-time communication with IntentForge via MQTT:
1. Publishing intents
2. Subscribing to events
3. Real-time device control (IoT)
4. Live data streaming

Requirements:
    pip install paho-mqtt

Usage:
    python mqtt_realtime_example.py

    # Or specific examples:
    python mqtt_realtime_example.py --subscribe
    python mqtt_realtime_example.py --publish
    python mqtt_realtime_example.py --iot
"""

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import paho.mqtt.client as mqtt

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class MQTTConfig:
    """MQTT Configuration"""

    host: str = "localhost"
    port: int = 1883
    client_id: str = field(default_factory=lambda: f"python-{int(time.time())}")
    topic_prefix: str = "intentforge"
    keepalive: int = 60


class IntentForgeMQTT:
    """
    MQTT client for real-time IntentForge communication.

    Example:
        client = IntentForgeMQTT()
        client.connect()

        # Subscribe to events
        client.on("device:state", lambda data: print(f"Device changed: {data}"))

        # Send intent
        result = client.send_intent("Turn on living room lights")

        # Publish to topic
        client.publish("devices/light/living", {"state": "on"})
    """

    def __init__(self, config: MQTTConfig | None = None):
        self.config = config or MQTTConfig()
        self.client = mqtt.Client(client_id=self.config.client_id)
        self.connected = False
        self.pending_requests: dict[str, dict] = {}
        self.event_handlers: dict[str, list[Callable]] = {}

        # Setup callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    def connect(self) -> bool:
        """Connect to MQTT broker"""
        try:
            self.client.connect(self.config.host, self.config.port, self.config.keepalive)
            self.client.loop_start()

            # Wait for connection
            timeout = 5
            start = time.time()
            while not self.connected and (time.time() - start) < timeout:
                time.sleep(0.1)

            return self.connected

        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def disconnect(self):
        """Disconnect from broker"""
        self.client.loop_stop()
        self.client.disconnect()
        self.connected = False

    def _on_connect(self, client, userdata, flags, rc):
        """Handle connection"""
        if rc == 0:
            self.connected = True
            print(f"✅ Connected to MQTT broker at {self.config.host}:{self.config.port}")

            # Subscribe to response topics
            self.client.subscribe(f"{self.config.topic_prefix}/+/response/{self.config.client_id}")
            self.client.subscribe(f"{self.config.topic_prefix}/events/#")
        else:
            print(f"❌ Connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        """Handle disconnection"""
        self.connected = False
        print(f"Disconnected from MQTT broker (rc={rc})")

    def _on_message(self, client, userdata, msg):
        """Handle incoming messages"""
        try:
            payload = json.loads(msg.payload.decode())
            topic = msg.topic

            # Handle pending request responses
            if "request_id" in payload and payload["request_id"] in self.pending_requests:
                request = self.pending_requests.pop(payload["request_id"])
                request["response"] = payload
                request["event"].set()
                return

            # Handle events
            # Extract event type from topic
            if "/events/" in topic:
                event_type = topic.split("/events/")[-1]
                self._emit(event_type, payload)

            # Also emit for specific topic patterns
            for pattern, handlers in self.event_handlers.items():
                if self._topic_matches(topic, pattern):
                    for handler in handlers:
                        try:
                            handler(payload)
                        except Exception as e:
                            print(f"Handler error: {e}")

        except json.JSONDecodeError:
            print(f"Invalid JSON in message: {msg.payload}")
        except Exception as e:
            print(f"Message handling error: {e}")

    def _topic_matches(self, topic: str, pattern: str) -> bool:
        """Check if topic matches pattern (supports # and + wildcards)"""
        topic_parts = topic.split("/")
        pattern_parts = pattern.split("/")

        for i, part in enumerate(pattern_parts):
            if part == "#":
                return True
            if i >= len(topic_parts):
                return False
            if part != "+" and part != topic_parts[i]:
                return False

        return len(topic_parts) == len(pattern_parts)

    def _emit(self, event: str, data: Any):
        """Emit event to handlers"""
        if event in self.event_handlers:
            for handler in self.event_handlers[event]:
                try:
                    handler(data)
                except Exception as e:
                    print(f"Event handler error: {e}")

    # =========================================================================
    # Public API
    # =========================================================================

    def on(self, event: str, handler: Callable) -> "IntentForgeMQTT":
        """
        Subscribe to an event.

        Args:
            event: Event name or topic pattern
            handler: Callback function

        Example:
            client.on("device:state", lambda data: print(data))
            client.on("sensor/+/temperature", handle_temp)
        """
        if event not in self.event_handlers:
            self.event_handlers[event] = []
            # Subscribe to topic if it looks like one
            if "/" in event or event.startswith(self.config.topic_prefix):
                self.client.subscribe(event)

        self.event_handlers[event].append(handler)
        return self

    def off(self, event: str, handler: Callable | None = None) -> "IntentForgeMQTT":
        """Unsubscribe from an event"""
        if event in self.event_handlers:
            if handler:
                self.event_handlers[event].remove(handler)
            else:
                del self.event_handlers[event]
        return self

    def publish(self, topic: str, payload: Any, qos: int = 1) -> bool:
        """
        Publish message to topic.

        Args:
            topic: MQTT topic
            payload: Message payload (will be JSON encoded)
            qos: Quality of Service (0, 1, or 2)
        """
        if not self.connected:
            print("Not connected to broker")
            return False

        message = json.dumps(payload) if isinstance(payload, (dict, list)) else str(payload)
        result = self.client.publish(topic, message, qos=qos)
        return result.rc == mqtt.MQTT_ERR_SUCCESS

    def send_intent(
        self,
        description: str,
        intent_type: str = "workflow",
        context: dict | None = None,
        timeout: float = 30.0,
    ) -> dict | None:
        """
        Send intent and wait for response.

        Args:
            description: Intent description
            intent_type: Type of intent
            context: Additional context
            timeout: Response timeout in seconds

        Returns:
            Response dict or None if timeout
        """
        request_id = str(uuid.uuid4())

        payload = {
            "request_id": request_id,
            "description": description,
            "intent_type": intent_type,
            "context": context or {},
        }

        # Setup pending request
        event = asyncio.Event() if asyncio.get_event_loop().is_running() else None
        self.pending_requests[request_id] = {
            "event": event
            or type("Event", (), {"set": lambda: None, "wait": lambda t: time.sleep(t)})(),
            "response": None,
        }

        # Publish request
        topic = f"{self.config.topic_prefix}/intent/request/{self.config.client_id}"
        self.publish(topic, payload)

        # Wait for response
        start = time.time()
        while (time.time() - start) < timeout:
            if request_id not in self.pending_requests:
                break
            if self.pending_requests.get(request_id, {}).get("response"):
                return self.pending_requests.pop(request_id)["response"]
            time.sleep(0.1)

        # Timeout
        self.pending_requests.pop(request_id, None)
        return None

    def send_device_command(self, device_id: str, command: str, params: dict | None = None) -> bool:
        """
        Send command to IoT device.

        Args:
            device_id: Device identifier
            command: Command name (on, off, set, etc.)
            params: Command parameters
        """
        topic = f"{self.config.topic_prefix}/devices/{device_id}/command"
        payload = {"command": command, "params": params or {}, "timestamp": time.time()}
        return self.publish(topic, payload)


# =============================================================================
# Example Functions
# =============================================================================


def example_basic_pubsub():
    """Basic publish/subscribe example"""
    print("\n" + "=" * 60)
    print("1. Basic Pub/Sub")
    print("=" * 60)

    client = IntentForgeMQTT()

    if not client.connect():
        print("❌ Could not connect to MQTT broker")
        print("   Make sure Mosquitto is running: docker-compose up -d mqtt")
        return

    received_messages = []

    # Subscribe to test topic
    def on_test_message(data):
        print(f"📨 Received: {data}")
        received_messages.append(data)

    client.on("intentforge/test/#", on_test_message)

    # Publish test messages
    print("\n[Publishing messages...]")
    for i in range(3):
        client.publish(
            f"intentforge/test/message{i}",
            {"index": i, "content": f"Test message {i}", "timestamp": time.time()},
        )
        time.sleep(0.5)

    # Wait for messages
    time.sleep(1)
    print(f"\n✅ Received {len(received_messages)} messages")

    client.disconnect()


def example_intent_processing():
    """Send intents via MQTT"""
    print("\n" + "=" * 60)
    print("2. Intent Processing via MQTT")
    print("=" * 60)

    client = IntentForgeMQTT()

    if not client.connect():
        print("❌ Could not connect to MQTT broker")
        return

    intents = [
        "Return a list of 5 mock users with id, name, and email",
        "Calculate fibonacci of 10",
        "Generate a random password with 16 characters",
    ]

    for description in intents:
        print(f"\n[Intent] {description}")
        print("-" * 40)

        result = client.send_intent(description, timeout=10)

        if result:
            print(f"Status: {'✅ Success' if result.get('success') else '❌ Failed'}")
            if result.get("result"):
                print(f"Result: {json.dumps(result['result'], indent=2)[:200]}...")
        else:
            print("⏱️ Timeout - no response received")
            print("   Make sure IntentForge worker is running")

    client.disconnect()


def example_iot_simulation():
    """IoT device simulation"""
    print("\n" + "=" * 60)
    print("3. IoT Device Simulation")
    print("=" * 60)

    client = IntentForgeMQTT()

    if not client.connect():
        print("❌ Could not connect to MQTT broker")
        return

    # Simulate device state
    devices = {
        "living_light": {"state": "off", "brightness": 0},
        "thermostat": {"state": "on", "temperature": 22},
        "door_sensor": {"state": "closed"},
    }

    # Subscribe to device commands
    def handle_device_command(data):
        device_id = data.get("device_id")
        command = data.get("command")
        params = data.get("params", {})

        print(f"📡 Command received: {device_id} -> {command}")

        if device_id in devices:
            if command == "on":
                devices[device_id]["state"] = "on"
            elif command == "off":
                devices[device_id]["state"] = "off"
            elif command == "set":
                devices[device_id].update(params)

            # Publish state update
            client.publish(f"intentforge/devices/{device_id}/state", devices[device_id])
            print(f"   New state: {devices[device_id]}")

    client.on("intentforge/devices/+/command", handle_device_command)

    # Simulate sending commands
    print("\n[Simulating device commands...]")

    commands = [
        ("living_light", "on", {}),
        ("living_light", "set", {"brightness": 75}),
        ("thermostat", "set", {"temperature": 24}),
        ("living_light", "off", {}),
    ]

    for device_id, command, params in commands:
        print(f"\n→ Sending: {device_id}.{command}({params})")
        client.publish(
            f"intentforge/devices/{device_id}/command",
            {"device_id": device_id, "command": command, "params": params},
        )
        time.sleep(1)

    print("\n[Final device states]")
    for device_id, state in devices.items():
        print(f"  {device_id}: {state}")

    client.disconnect()


def example_sensor_streaming():
    """Real-time sensor data streaming"""
    print("\n" + "=" * 60)
    print("4. Sensor Data Streaming")
    print("=" * 60)

    client = IntentForgeMQTT()

    if not client.connect():
        print("❌ Could not connect to MQTT broker")
        return

    # Subscribe to sensor data
    sensor_data = []

    def handle_sensor_data(data):
        sensor_data.append(data)
        print(f"📊 Sensor: {data.get('sensor')} = {data.get('value')}{data.get('unit', '')}")

    client.on("intentforge/sensors/#", handle_sensor_data)

    # Simulate sensor readings
    print("\n[Simulating sensor readings...]")

    import random

    sensors = [
        ("temperature", 20, 25, "°C"),
        ("humidity", 40, 60, "%"),
        ("pressure", 1010, 1020, "hPa"),
        ("light", 200, 800, "lux"),
    ]

    for _ in range(5):
        for sensor_name, min_val, max_val, unit in sensors:
            value = round(random.uniform(min_val, max_val), 1)
            client.publish(
                f"intentforge/sensors/{sensor_name}",
                {"sensor": sensor_name, "value": value, "unit": unit, "timestamp": time.time()},
            )
        time.sleep(1)

    print(f"\n✅ Streamed {len(sensor_data)} sensor readings")

    client.disconnect()


def example_event_driven():
    """Event-driven architecture example"""
    print("\n" + "=" * 60)
    print("5. Event-Driven Architecture")
    print("=" * 60)

    client = IntentForgeMQTT()

    if not client.connect():
        print("❌ Could not connect to MQTT broker")
        return

    # Event handlers
    def on_order_created(data):
        print(f"🛒 New order: #{data.get('order_id')} - {data.get('total')} PLN")
        # Trigger follow-up events
        client.publish(
            "intentforge/events/inventory/check",
            {"order_id": data.get("order_id"), "items": data.get("items", [])},
        )

    def on_inventory_check(data):
        print(f"📦 Checking inventory for order #{data.get('order_id')}")
        # Simulate inventory check
        client.publish(
            "intentforge/events/shipping/prepare",
            {"order_id": data.get("order_id"), "status": "ready"},
        )

    def on_shipping_prepare(data):
        print(f"🚚 Preparing shipment for order #{data.get('order_id')}")

    # Subscribe to events
    client.on("intentforge/events/order/created", on_order_created)
    client.on("intentforge/events/inventory/check", on_inventory_check)
    client.on("intentforge/events/shipping/prepare", on_shipping_prepare)

    # Simulate order creation
    print("\n[Simulating order flow...]")

    client.publish(
        "intentforge/events/order/created",
        {
            "order_id": 12345,
            "customer": "jan@example.com",
            "items": [{"product": "Laptop", "qty": 1}, {"product": "Mouse", "qty": 2}],
            "total": 4599.99,
        },
    )

    # Wait for event chain
    time.sleep(2)

    print("\n✅ Event chain completed")

    client.disconnect()


def main():
    """Run all examples"""
    print("\n" + "#" * 60)
    print("  IntentForge MQTT Real-time Examples")
    print("#" * 60)

    example_basic_pubsub()
    example_intent_processing()
    example_iot_simulation()
    example_sensor_streaming()
    example_event_driven()

    print("\n" + "=" * 60)
    print("✅ All MQTT examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IntentForge MQTT Examples")
    parser.add_argument("--pubsub", action="store_true", help="Run pub/sub example")
    parser.add_argument("--intent", action="store_true", help="Run intent example")
    parser.add_argument("--iot", action="store_true", help="Run IoT example")
    parser.add_argument("--sensors", action="store_true", help="Run sensor streaming")
    parser.add_argument("--events", action="store_true", help="Run event-driven example")

    args = parser.parse_args()

    if args.pubsub:
        example_basic_pubsub()
    elif args.intent:
        example_intent_processing()
    elif args.iot:
        example_iot_simulation()
    elif args.sensors:
        example_sensor_streaming()
    elif args.events:
        example_event_driven()
    else:
        main()

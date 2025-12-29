"""
IntentForge AutoWorker Agent

Handles multi-role agent tasks:
- Developer: Code generation and fixing
- Reviewer: Code review and quality checks
- QA: Test generation and execution
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

class WorkerRole(Enum):
    DEVELOPER = "developer"
    REVIEWER = "reviewer"
    QA = "qa"
    DEVOPS = "devops"

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class AgentTask:
    id: str
    role: WorkerRole
    description: str
    code: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    logs: List[str] = field(default_factory=list)

class WorkerOrchestrator:
    def __init__(self):
        self.tasks: Dict[str, AgentTask] = {}
        self.subscribers: Dict[str, List[asyncio.Queue]] = {}

    def create_task(self, role: str, description: str, code: Optional[str] = None) -> AgentTask:
        task_id = str(uuid.uuid4())[:8]
        try:
            worker_role = WorkerRole(role.lower())
        except ValueError:
            worker_role = WorkerRole.DEVELOPER

        task = AgentTask(
            id=task_id,
            role=worker_role,
            description=description,
            code=code
        )
        self.tasks[task_id] = task
        self.subscribers[task_id] = []
        return task

    def get_task(self, task_id: str) -> Optional[AgentTask]:
        return self.tasks.get(task_id)

    async def subscribe(self, task_id: str) -> asyncio.Queue:
        if task_id not in self.subscribers:
            self.subscribers[task_id] = []
        queue = asyncio.Queue()
        self.subscribers[task_id].append(queue)
        return queue

    async def unsubscribe(self, task_id: str, queue: asyncio.Queue):
        if task_id in self.subscribers and queue in self.subscribers[task_id]:
            self.subscribers[task_id].remove(queue)

    async def emit_log(self, task_id: str, message: str, level: str = "info"):
        if task_id in self.tasks:
            self.tasks[task_id].logs.append(message)
            if task_id in self.subscribers:
                data = {
                    "type": "log",
                    "level": level,
                    "message": message,
                    "task_id": task_id
                }
                for queue in self.subscribers[task_id]:
                    await queue.put(data)

    async def emit_status(self, task_id: str, status: TaskStatus):
        if task_id in self.tasks:
            self.tasks[task_id].status = status
            if task_id in self.subscribers:
                data = {
                    "type": "status",
                    "status": status.value,
                    "task_id": task_id
                }
                for queue in self.subscribers[task_id]:
                    await queue.put(data)

    async def emit_result(self, task_id: str, result: str):
        if task_id in self.tasks:
            self.tasks[task_id].result = result
            self.tasks[task_id].status = TaskStatus.COMPLETED
            if task_id in self.subscribers:
                data = {
                    "type": "result",
                    "result": result,
                    "task_id": task_id
                }
                for queue in self.subscribers[task_id]:
                    await queue.put(data)

def create_agent_routes(orchestrator: WorkerOrchestrator) -> APIRouter:
    router = APIRouter(prefix="/api/agent")

    @router.websocket("/ws/{task_id}")
    async def agent_websocket(websocket: WebSocket, task_id: str):
        await websocket.accept()
        queue = await orchestrator.subscribe(task_id)
        try:
            task = orchestrator.get_task(task_id)
            if task:
                # Send history
                for log in task.logs:
                    await websocket.send_json({"type": "log", "message": log, "task_id": task_id})
                if task.result:
                    await websocket.send_json({"type": "result", "result": task.result, "task_id": task_id})

            while True:
                data = await queue.get()
                await websocket.send_json(data)
        except WebSocketDisconnect:
            pass
        finally:
            await orchestrator.unsubscribe(task_id, queue)

    return router

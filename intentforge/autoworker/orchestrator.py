"""
WorkerOrchestrator - Task routing and worker management.

Coordinates multiple workers:
- Routes tasks to appropriate workers
- Manages worker lifecycle
- Handles priorities and queues
- Resource management
"""

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .base import BaseWorker, Task, TaskStatus, WorkerResult, WorkerRole

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Task routing strategies"""

    ROLE_BASED = "role_based"
    CAPABILITY_BASED = "capability_based"
    LOAD_BALANCED = "load_balanced"
    ROUND_ROBIN = "round_robin"


@dataclass
class TaskQueue:
    """Priority queue for tasks"""

    tasks: list[Task] = field(default_factory=list)

    def push(self, task: Task):
        """Add task to queue (sorted by priority)"""
        self.tasks.append(task)
        self.tasks.sort(key=lambda t: (t.priority.value, t.created_at))

    def pop(self) -> Task | None:
        """Get highest priority task"""
        return self.tasks.pop(0) if self.tasks else None

    def peek(self) -> Task | None:
        """View highest priority task without removing"""
        return self.tasks[0] if self.tasks else None

    def remove(self, task_id: str) -> bool:
        """Remove task by ID"""
        for i, task in enumerate(self.tasks):
            if task.id == task_id:
                self.tasks.pop(i)
                return True
        return False

    def __len__(self) -> int:
        return len(self.tasks)


@dataclass
class WorkerPool:
    """Pool of workers by role"""

    workers: dict[WorkerRole, list[BaseWorker]] = field(default_factory=dict)

    def register(self, worker: BaseWorker):
        """Register a worker"""
        role = worker.role
        if role not in self.workers:
            self.workers[role] = []
        self.workers[role].append(worker)
        logger.info(f"Registered worker: {worker.name} ({role.value})")

    def unregister(self, worker: BaseWorker):
        """Unregister a worker"""
        role = worker.role
        if role in self.workers:
            self.workers[role] = [w for w in self.workers[role] if w != worker]

    def get_by_role(self, role: WorkerRole) -> list[BaseWorker]:
        """Get all workers of a role"""
        return self.workers.get(role, [])

    def get_available(self, role: WorkerRole) -> BaseWorker | None:
        """Get an available worker of a role"""
        workers = self.get_by_role(role)
        for worker in workers:
            # Check if worker has capacity
            if len(worker._running_tasks) < worker.config.max_concurrent_tasks:
                return worker
        return None

    def all_workers(self) -> list[BaseWorker]:
        """Get all workers"""
        result = []
        for workers in self.workers.values():
            result.extend(workers)
        return result


class WorkerOrchestrator:
    """
    Orchestrates task execution across multiple workers.

    Features:
    - Task routing to appropriate workers
    - Priority queue management
    - Worker lifecycle management
    - Resource and cost tracking
    - Event-driven task triggering
    """

    # Task type to role mapping
    TASK_ROLE_MAPPING = {
        "write_code": WorkerRole.DEVELOPER,
        "review_code": WorkerRole.DEVELOPER,
        "refactor": WorkerRole.DEVELOPER,
        "debug": WorkerRole.DEVELOPER,
        "generate_tests": WorkerRole.QA,
        "run_tests": WorkerRole.QA,
        "security_scan": WorkerRole.QA,
        "create_dockerfile": WorkerRole.DEVOPS,
        "deploy": WorkerRole.DEVOPS,
        "create_pipeline": WorkerRole.DEVOPS,
        "generate_sql": WorkerRole.ANALYST,
        "analyze_data": WorkerRole.ANALYST,
        "answer_question": WorkerRole.SUPPORT,
        "search_docs": WorkerRole.SUPPORT,
    }

    def __init__(
        self,
        strategy: RoutingStrategy = RoutingStrategy.ROLE_BASED,
        max_queue_size: int = 1000,
    ):
        self.strategy = strategy
        self.max_queue_size = max_queue_size
        self.pool = WorkerPool()
        self.queue = TaskQueue()
        self._is_running = False
        self._processing_task: asyncio.Task | None = None
        self._completed_tasks: list[tuple[Task, WorkerResult]] = []
        self._cost_tracker: dict[str, float] = {}

    def register_worker(self, worker: BaseWorker):
        """Register a worker with the orchestrator"""
        worker.setup()
        self.pool.register(worker)

    def unregister_worker(self, worker: BaseWorker):
        """Unregister a worker"""
        worker.teardown()
        self.pool.unregister(worker)

    async def submit(self, task: Task) -> str:
        """
        Submit a task for execution.

        Args:
            task: Task to execute

        Returns:
            Task ID
        """
        if len(self.queue) >= self.max_queue_size:
            raise RuntimeError("Task queue is full")

        task.status = TaskStatus.PENDING
        self.queue.push(task)
        logger.info(f"Task submitted: {task.id} ({task.type}) priority={task.priority.value}")

        return task.id

    async def execute(self, task: Task) -> WorkerResult:
        """
        Execute a task immediately.

        Args:
            task: Task to execute

        Returns:
            WorkerResult
        """
        # Route to appropriate worker
        worker = await self._route_task(task)
        if not worker:
            return WorkerResult(
                success=False,
                task_id=task.id,
                error=f"No available worker for task type: {task.type}",
            )

        # Execute
        result = await worker.run(task)

        # Track
        self._completed_tasks.append((task, result))

        return result

    async def start(self):
        """Start the orchestrator processing loop"""
        if self._is_running:
            return

        self._is_running = True
        self._processing_task = asyncio.create_task(self._process_loop())
        logger.info("Orchestrator started")

    async def stop(self):
        """Stop the orchestrator"""
        self._is_running = False
        if self._processing_task:
            self._processing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._processing_task
        logger.info("Orchestrator stopped")

    async def _process_loop(self):
        """Main processing loop"""
        while self._is_running:
            task = self.queue.pop()
            if task:
                try:
                    await self.execute(task)
                except Exception as e:
                    logger.exception(f"Task {task.id} failed: {e}")
            else:
                await asyncio.sleep(0.1)

    async def _route_task(self, task: Task) -> BaseWorker | None:
        """Route task to appropriate worker"""
        if self.strategy == RoutingStrategy.ROLE_BASED:
            return await self._route_by_role(task)
        elif self.strategy == RoutingStrategy.CAPABILITY_BASED:
            return await self._route_by_capability(task)
        elif self.strategy == RoutingStrategy.LOAD_BALANCED:
            return await self._route_load_balanced(task)
        else:
            return await self._route_round_robin(task)

    async def _route_by_role(self, task: Task) -> BaseWorker | None:
        """Route based on task type to role mapping"""
        role = self.TASK_ROLE_MAPPING.get(task.type)
        if not role:
            # Default to developer for unknown tasks
            role = WorkerRole.DEVELOPER

        return self.pool.get_available(role)

    async def _route_by_capability(self, task: Task) -> BaseWorker | None:
        """Route based on worker capabilities"""
        for worker in self.pool.all_workers():
            if worker.can_handle(task):
                if len(worker._running_tasks) < worker.config.max_concurrent_tasks:
                    return worker
        return None

    async def _route_load_balanced(self, task: Task) -> BaseWorker | None:
        """Route to least loaded worker that can handle the task"""
        candidates = []
        for worker in self.pool.all_workers():
            if worker.can_handle(task):
                load = len(worker._running_tasks) / worker.config.max_concurrent_tasks
                candidates.append((load, worker))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
        return None

    async def _route_round_robin(self, task: Task) -> BaseWorker | None:
        """Simple round-robin routing"""
        workers = [w for w in self.pool.all_workers() if w.can_handle(task)]
        if workers:
            # Rotate based on completed tasks count
            idx = len(self._completed_tasks) % len(workers)
            return workers[idx]
        return None

    async def get_task_status(self, task_id: str) -> dict[str, Any]:
        """Get status of a task"""
        # Check queue
        for task in self.queue.tasks:
            if task.id == task_id:
                return {
                    "id": task_id,
                    "status": task.status.value,
                    "priority": task.priority.value,
                    "queued": True,
                }

        # Check running
        for worker in self.pool.all_workers():
            if task_id in worker._running_tasks:
                task = worker._running_tasks[task_id]
                return {
                    "id": task_id,
                    "status": task.status.value,
                    "worker": worker.name,
                    "running": True,
                }

        # Check completed
        for task, result in self._completed_tasks:
            if task.id == task_id:
                return {
                    "id": task_id,
                    "status": task.status.value,
                    "success": result.success,
                    "completed": True,
                    "duration_ms": result.duration_ms,
                }

        return {"id": task_id, "status": "not_found"}

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task"""
        return self.queue.remove(task_id)

    async def health_check(self) -> dict[str, Any]:
        """Get orchestrator health status"""
        worker_health = []
        for worker in self.pool.all_workers():
            worker_health.append(await worker.health_check())

        return {
            "healthy": self._is_running,
            "queue_size": len(self.queue),
            "completed_tasks": len(self._completed_tasks),
            "workers": worker_health,
            "strategy": self.strategy.value,
        }

    def get_metrics(self) -> dict[str, Any]:
        """Get orchestrator metrics"""
        success_count = sum(1 for _, r in self._completed_tasks if r.success)
        total_duration = sum(r.duration_ms for _, r in self._completed_tasks)

        return {
            "total_tasks": len(self._completed_tasks),
            "success_rate": success_count / len(self._completed_tasks)
            if self._completed_tasks
            else 0,
            "average_duration_ms": total_duration / len(self._completed_tasks)
            if self._completed_tasks
            else 0,
            "queue_size": len(self.queue),
            "worker_count": len(self.pool.all_workers()),
        }

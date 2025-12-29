"""
BaseWorker - Abstract base class for all AutoWorker roles.

All workers (Developer, QA, DevOps, Analyst, Support) inherit from this class.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class WorkerRole(Enum):
    """Worker role types"""

    DEVELOPER = "developer"
    QA = "qa"
    DEVOPS = "devops"
    ANALYST = "analyst"
    SUPPORT = "support"


class TaskPriority(Enum):
    """Task priority levels"""

    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class TaskStatus(Enum):
    """Task execution status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkerConfig:
    """Configuration for a worker"""

    role: WorkerRole
    name: str = ""
    max_concurrent_tasks: int = 5
    timeout_seconds: int = 300
    retry_count: int = 3
    auto_rollback: bool = True
    require_approval_for: list[str] = field(default_factory=list)
    budget_limit: float = 0.0  # 0 = unlimited
    custom_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """Task to be executed by a worker"""

    id: str
    type: str
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    params: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    assigned_worker: str | None = None
    parent_task_id: str | None = None
    subtasks: list[str] = field(default_factory=list)


@dataclass
class WorkerResult:
    """Result of a worker task execution"""

    success: bool
    task_id: str
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    rollback_available: bool = False
    rollback_id: str | None = None
    artifacts: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)


class BaseWorker(ABC):
    """
    Abstract base class for all AutoWorker roles.

    Provides common functionality:
    - Task execution with timeout and retry
    - Pre/post validation hooks
    - Rollback support
    - Audit logging
    - Health checks
    """

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.name = config.name or f"{config.role.value}_worker"
        self._running_tasks: dict[str, Task] = {}
        self._completed_tasks: list[str] = []
        self._is_healthy = True
        self._task_semaphore = asyncio.Semaphore(config.max_concurrent_tasks)

    @property
    def role(self) -> WorkerRole:
        return self.config.role

    @abstractmethod
    async def execute(self, task: Task) -> WorkerResult:
        """
        Execute a task. Must be implemented by subclasses.

        Args:
            task: The task to execute

        Returns:
            WorkerResult with execution outcome
        """
        pass

    @abstractmethod
    def can_handle(self, task: Task) -> bool:
        """
        Check if this worker can handle the given task.

        Args:
            task: The task to check

        Returns:
            True if worker can handle this task type
        """
        pass

    async def run(self, task: Task) -> WorkerResult:
        """
        Run a task with full lifecycle management.

        Includes:
        - Concurrency limiting
        - Pre-validation
        - Timeout handling
        - Retry logic
        - Post-validation
        - Audit logging
        """
        async with self._task_semaphore:
            return await self._run_with_retry(task)

    async def _run_with_retry(self, task: Task) -> WorkerResult:
        """Execute task with retry logic"""
        last_error = None

        for attempt in range(self.config.retry_count + 1):
            try:
                result = await self._run_single(task, attempt)
                if result.success:
                    return result
                last_error = result.error
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Worker {self.name} task {task.id} attempt {attempt + 1} failed: {e}"
                )

            if attempt < self.config.retry_count:
                await asyncio.sleep(2**attempt)  # Exponential backoff

        return WorkerResult(
            success=False,
            task_id=task.id,
            error=f"Failed after {self.config.retry_count + 1} attempts: {last_error}",
        )

    async def _run_single(self, task: Task, attempt: int) -> WorkerResult:
        """Execute a single task attempt"""
        start_time = datetime.utcnow()

        # Track running task
        task.status = TaskStatus.RUNNING
        task.started_at = start_time.isoformat()
        task.assigned_worker = self.name
        self._running_tasks[task.id] = task

        try:
            # Pre-validation
            validation_result = await self.pre_validate(task)
            if not validation_result["valid"]:
                return WorkerResult(
                    success=False,
                    task_id=task.id,
                    error=f"Pre-validation failed: {validation_result.get('reason')}",
                )

            # Check if approval required
            if await self._requires_approval(task):
                logger.info(f"Task {task.id} requires approval")
                # In real implementation, would wait for approval

            # Execute with timeout
            result = await asyncio.wait_for(
                self.execute(task),
                timeout=self.config.timeout_seconds,
            )

            # Post-validation
            if result.success:
                post_valid = await self.post_validate(task, result)
                if not post_valid["valid"]:
                    # Attempt rollback
                    if self.config.auto_rollback and result.rollback_available:
                        await self.rollback(result.rollback_id)
                    result.success = False
                    result.error = f"Post-validation failed: {post_valid.get('reason')}"

            # Update task status
            task.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
            task.completed_at = datetime.utcnow().isoformat()

            # Calculate duration
            result.duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Audit log
            await self._audit_log(task, result, attempt)

            return result

        except TimeoutError:
            task.status = TaskStatus.FAILED
            return WorkerResult(
                success=False,
                task_id=task.id,
                error=f"Task timed out after {self.config.timeout_seconds}s",
            )
        except Exception as e:
            task.status = TaskStatus.FAILED
            logger.exception(f"Worker {self.name} task {task.id} error")
            return WorkerResult(
                success=False,
                task_id=task.id,
                error=str(e),
            )
        finally:
            self._running_tasks.pop(task.id, None)
            self._completed_tasks.append(task.id)

    async def pre_validate(self, task: Task) -> dict[str, Any]:
        """
        Pre-execution validation hook.
        Override in subclasses for custom validation.
        """
        return {"valid": True}

    async def post_validate(self, task: Task, result: WorkerResult) -> dict[str, Any]:
        """
        Post-execution validation hook.
        Override in subclasses for custom validation.
        """
        return {"valid": True}

    async def rollback(self, rollback_id: str | None) -> bool:
        """
        Rollback a task execution.
        Override in subclasses for custom rollback logic.
        """
        logger.info(f"Rollback requested for {rollback_id}")
        return True

    async def _requires_approval(self, task: Task) -> bool:
        """Check if task requires human approval"""
        return task.type in self.config.require_approval_for

    async def _audit_log(self, task: Task, result: WorkerResult, attempt: int):
        """Log task execution for audit trail"""
        logger.info(
            f"AUDIT: worker={self.name} task={task.id} type={task.type} "
            f"attempt={attempt + 1} success={result.success} "
            f"duration_ms={result.duration_ms:.2f}"
        )

    async def health_check(self) -> dict[str, Any]:
        """Check worker health status"""
        return {
            "healthy": self._is_healthy,
            "name": self.name,
            "role": self.role.value,
            "running_tasks": len(self._running_tasks),
            "completed_tasks": len(self._completed_tasks),
            "max_concurrent": self.config.max_concurrent_tasks,
        }

    def setup(self):
        """Initialize worker resources. Override in subclasses."""
        pass

    def teardown(self):
        """Cleanup worker resources. Override in subclasses."""
        pass

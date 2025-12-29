"""
IntentForge Autonomous Workflow Engine

Multi-step, proactive LLM workflows that:
- Chain multiple LLM requests
- Use previously generated modules
- Execute code in sandbox
- Self-correct and retry on failures
- Build reusable module library
"""

import asyncio
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class WorkflowStepType(Enum):
    """Types of workflow steps"""

    LLM_GENERATE = "llm_generate"  # Generate code/text with LLM
    EXECUTE_CODE = "execute_code"  # Run code in sandbox
    EXECUTE_MODULE = "execute_module"  # Call existing module
    CREATE_MODULE = "create_module"  # Create new reusable module
    CONDITION = "condition"  # Conditional branching
    LOOP = "loop"  # Loop over items
    PARALLEL = "parallel"  # Run steps in parallel
    HUMAN_INPUT = "human_input"  # Wait for human input


@dataclass
class WorkflowStep:
    """A single step in an autonomous workflow"""

    id: str
    type: WorkflowStepType
    name: str
    config: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    retry_count: int = 3
    timeout: float = 60.0
    on_error: str = "retry"  # retry, skip, abort, fallback


@dataclass
class WorkflowContext:
    """Shared context across workflow steps"""

    variables: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)
    created_modules: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)


@dataclass
class WorkflowResult:
    """Result of workflow execution"""

    success: bool
    steps_executed: int
    steps_failed: int
    final_result: Any = None
    context: WorkflowContext = None
    execution_time_ms: float = 0


class AutonomousWorkflow:
    """
    Autonomous workflow engine for multi-step LLM tasks.

    Example DSL:
    ```
    workflow "data_processor" do
        step generate_code with llm do
            prompt = "Create a CSV parser"
            save_as = $parser_code
        end

        step test_code with execute do
            code = $parser_code
            input = {"file": "test.csv"}
        end

        step create_module if $test_result.success do
            name = "csv_parser"
            code = $parser_code
        end
    end
    ```
    """

    def __init__(self, name: str = "unnamed"):
        self.name = name
        self.steps: list[WorkflowStep] = []
        self.context = WorkflowContext()
        self.hooks: dict[str, list[Callable]] = {
            "before_step": [],
            "after_step": [],
            "on_error": [],
            "on_complete": [],
        }

    def add_step(self, step: WorkflowStep) -> "AutonomousWorkflow":
        """Add a step to the workflow"""
        self.steps.append(step)
        return self

    def set_variable(self, name: str, value: Any) -> "AutonomousWorkflow":
        """Set a context variable"""
        self.context.variables[name] = value
        return self

    def on(self, event: str, callback: Callable) -> "AutonomousWorkflow":
        """Register event hook"""
        if event in self.hooks:
            self.hooks[event].append(callback)
        return self

    async def execute(self) -> WorkflowResult:
        """Execute the workflow"""
        import time

        start_time = time.time()

        result = WorkflowResult(
            success=True,
            steps_executed=0,
            steps_failed=0,
            context=self.context,
        )

        executed_steps = set()

        for step in self.steps:
            # Check dependencies
            if not all(dep in executed_steps for dep in step.depends_on):
                logger.warning(f"Skipping step {step.id}: dependencies not met")
                continue

            # Execute hooks
            for hook in self.hooks["before_step"]:
                await self._call_hook(hook, step, self.context)

            # Execute step with retries
            step_result = None
            last_error = None

            for attempt in range(step.retry_count):
                try:
                    step_result = await self._execute_step(step)
                    if step_result.get("success"):
                        break
                    last_error = step_result.get("error")
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"Step {step.id} attempt {attempt + 1} failed: {e}")

                if attempt < step.retry_count - 1:
                    await asyncio.sleep(1)  # Brief pause before retry

            # Store result
            self.context.results[step.id] = step_result
            self.context.history.append(
                {
                    "step": step.id,
                    "type": step.type.value,
                    "result": step_result,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            if step_result and step_result.get("success"):
                result.steps_executed += 1
                executed_steps.add(step.id)

                # Execute after hooks
                for hook in self.hooks["after_step"]:
                    await self._call_hook(hook, step, step_result, self.context)
            else:
                result.steps_failed += 1
                self.context.errors.append(
                    {
                        "step": step.id,
                        "error": last_error,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                # Execute error hooks
                for hook in self.hooks["on_error"]:
                    await self._call_hook(hook, step, last_error, self.context)

                # Handle error based on step config
                if step.on_error == "abort":
                    result.success = False
                    break

        # Execute completion hooks
        for hook in self.hooks["on_complete"]:
            await self._call_hook(hook, result, self.context)

        result.execution_time_ms = (time.time() - start_time) * 1000
        result.final_result = self.context.results.get(self.steps[-1].id) if self.steps else None

        return result

    async def _call_hook(self, hook: Callable, *args):
        """Call a hook function"""
        if asyncio.iscoroutinefunction(hook):
            await hook(*args)
        else:
            hook(*args)

    async def _execute_step(self, step: WorkflowStep) -> dict:
        """Execute a single workflow step"""
        handlers = {
            WorkflowStepType.LLM_GENERATE: self._step_llm_generate,
            WorkflowStepType.EXECUTE_CODE: self._step_execute_code,
            WorkflowStepType.EXECUTE_MODULE: self._step_execute_module,
            WorkflowStepType.CREATE_MODULE: self._step_create_module,
            WorkflowStepType.CONDITION: self._step_condition,
            WorkflowStepType.LOOP: self._step_loop,
            WorkflowStepType.PARALLEL: self._step_parallel,
        }

        handler = handlers.get(step.type)
        if not handler:
            return {"success": False, "error": f"Unknown step type: {step.type}"}

        return await handler(step)

    async def _step_llm_generate(self, step: WorkflowStep) -> dict:
        """Generate content with LLM"""
        from .services import ChatService

        config = step.config
        prompt = self._interpolate(config.get("prompt", ""))
        system = config.get("system", "You are a helpful assistant.")
        save_as = config.get("save_as")

        chat = ChatService()
        response = await chat.send(message=prompt, system=system)

        if response.get("success"):
            content = response.get("response", "")

            if save_as:
                self.context.variables[save_as] = content

            return {
                "success": True,
                "content": content,
                "tokens": response.get("total_tokens"),
            }

        return {"success": False, "error": response.get("error")}

    async def _step_execute_code(self, step: WorkflowStep) -> dict:
        """Execute code in sandbox"""
        from .code_runner import run_with_autofix

        config = step.config
        code = self._interpolate(config.get("code", ""))
        language = config.get("language", "python")
        save_as = config.get("save_as")

        result = await run_with_autofix(
            code=code,
            language=language,
            max_retries=step.retry_count,
            auto_install=True,
        )

        if save_as:
            self.context.variables[save_as] = result

        return result

    async def _step_execute_module(self, step: WorkflowStep) -> dict:
        """Execute an existing module"""
        from .modules import module_manager

        config = step.config
        module_name = self._interpolate(config.get("module", ""))
        action = config.get("action", "execute")
        data = config.get("data", {})
        save_as = config.get("save_as")

        # Interpolate data values
        data = {k: self._interpolate(v) if isinstance(v, str) else v for k, v in data.items()}

        result = await module_manager.execute_module(
            name=module_name,
            action=action,
            data=data,
        )

        if save_as:
            self.context.variables[save_as] = result

        return {
            "success": result.success,
            "result": result.result,
            "error": result.error,
        }

    async def _step_create_module(self, step: WorkflowStep) -> dict:
        """Create a new reusable module"""
        from .modules import module_manager

        config = step.config
        name = self._interpolate(config.get("name", ""))
        description = config.get("description", "")
        code = self._interpolate(config.get("code", ""))

        # If no code, generate from description
        if not code and description:
            module_info = await module_manager.create_from_llm(
                task_description=description,
                module_name=name,
            )
        else:
            module_info = await module_manager.create_module(
                name=name,
                description=description,
                code=code,
            )

        self.context.created_modules.append(module_info.name)

        # Optionally build and start
        if config.get("build", False):
            await module_manager.build_module(module_info.name)
        if config.get("start", False):
            await module_manager.start_module(module_info.name)

        return {
            "success": True,
            "module": module_info.name,
            "port": module_info.port,
        }

    async def _step_condition(self, step: WorkflowStep) -> dict:
        """Conditional branching"""
        config = step.config
        condition = config.get("condition", "")
        then_step = config.get("then")
        else_step = config.get("else")

        # Evaluate condition
        result = self._evaluate_condition(condition)

        if result and then_step:
            return await self._execute_step(then_step)
        elif not result and else_step:
            return await self._execute_step(else_step)

        return {"success": True, "condition_result": result}

    async def _step_loop(self, step: WorkflowStep) -> dict:
        """Loop over items"""
        config = step.config
        items_var = config.get("items", "")
        item_var = config.get("as", "item")
        body_steps = config.get("body", [])

        items = self.context.variables.get(items_var, [])
        results = []

        for item in items:
            self.context.variables[item_var] = item

            for body_step in body_steps:
                result = await self._execute_step(body_step)
                results.append(result)

        return {"success": True, "loop_results": results}

    async def _step_parallel(self, step: WorkflowStep) -> dict:
        """Execute steps in parallel"""
        config = step.config
        parallel_steps = config.get("steps", [])

        tasks = [self._execute_step(s) for s in parallel_steps]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            "success": all(
                isinstance(r, dict) and r.get("success")
                for r in results
                if not isinstance(r, Exception)
            ),
            "parallel_results": [
                r if not isinstance(r, Exception) else {"error": str(r)} for r in results
            ],
        }

    def _interpolate(self, value: str) -> str:
        """Interpolate variables in string"""
        if not isinstance(value, str):
            return value

        # Replace $variable with actual values
        def replace_var(match):
            var_name = match.group(1)
            parts = var_name.split(".")

            val = self.context.variables.get(parts[0])
            for part in parts[1:]:
                if isinstance(val, dict):
                    val = val.get(part)
                elif hasattr(val, part):
                    val = getattr(val, part)
                else:
                    return match.group(0)

            return str(val) if val is not None else match.group(0)

        return re.sub(r"\$(\w+(?:\.\w+)*)", replace_var, value)

    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate a condition expression"""
        # Simple condition evaluation
        condition = self._interpolate(condition)

        try:
            # Safe evaluation for simple conditions
            if condition.lower() in ("true", "1", "yes"):
                return True
            if condition.lower() in ("false", "0", "no", ""):
                return False

            # Check for comparison operators
            for op in ["==", "!=", ">=", "<=", ">", "<"]:
                if op in condition:
                    left, right = condition.split(op, 1)
                    left = left.strip()
                    right = right.strip()

                    # Try numeric comparison
                    try:
                        left_val = float(left)
                        right_val = float(right)
                    except ValueError:
                        left_val = left
                        right_val = right

                    if op == "==":
                        return left_val == right_val
                    elif op == "!=":
                        return left_val != right_val
                    elif op == ">=":
                        return left_val >= right_val
                    elif op == "<=":
                        return left_val <= right_val
                    elif op == ">":
                        return left_val > right_val
                    elif op == "<":
                        return left_val < right_val

            return bool(condition)
        except Exception:
            return False


class AutonomousAgent:
    """
    High-level autonomous agent that uses LLM to plan and execute workflows.

    The agent can:
    - Break down complex tasks into steps
    - Create and use modules
    - Self-correct on failures
    - Build a library of reusable code
    """

    def __init__(self):
        self.workflows: dict[str, AutonomousWorkflow] = {}
        self.task_history: list[dict] = []

    async def execute_task(
        self,
        task: str,
        context: dict | None = None,
        max_steps: int = 10,
    ) -> dict:
        """
        Execute a complex task autonomously.

        Args:
            task: Natural language task description
            context: Additional context/data
            max_steps: Maximum number of steps to execute

        Returns:
            Task execution result
        """
        from .modules import module_manager
        from .services import ChatService

        chat = ChatService()

        # Get available modules
        available_modules = module_manager.list_modules()
        modules_info = (
            "\n".join(
                [
                    f"- {m['name']}: {m['description']} (status: {m['status']})"
                    for m in available_modules
                ]
            )
            or "No modules available yet."
        )

        # Plan the task
        plan_prompt = f"""You are an autonomous AI agent. Plan how to complete this task:

TASK: {task}

AVAILABLE MODULES:
{modules_info}

CONTEXT:
{json.dumps(context or {}, indent=2)}

Create a step-by-step plan. For each step, specify:
1. Step type: llm_generate, execute_code, execute_module, create_module
2. Step details

Return as JSON array:
[
  {{"type": "llm_generate", "name": "step1", "prompt": "...", "save_as": "var1"}},
  {{"type": "execute_code", "name": "step2", "code": "...", "language": "python"}},
  ...
]
"""

        plan_response = await chat.send(
            message=plan_prompt,
            system="You are a task planner. Return only valid JSON, no explanations.",
        )

        if not plan_response.get("success"):
            return {"success": False, "error": "Failed to create plan"}

        # Parse plan
        try:
            plan_text = plan_response.get("response", "")
            # Extract JSON from response
            json_match = re.search(r"\[[\s\S]*\]", plan_text)
            if json_match:
                steps = json.loads(json_match.group())
            else:
                steps = json.loads(plan_text)
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Failed to parse plan: {e}"}

        # Create workflow from plan
        workflow = AutonomousWorkflow(name=f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        if context:
            for key, value in context.items():
                workflow.set_variable(key, value)

        for i, step_config in enumerate(steps[:max_steps]):
            step_type = WorkflowStepType[step_config.get("type", "llm_generate").upper()]
            step = WorkflowStep(
                id=step_config.get("name", f"step_{i}"),
                type=step_type,
                name=step_config.get("name", f"Step {i + 1}"),
                config=step_config,
            )
            workflow.add_step(step)

        # Execute workflow
        result = await workflow.execute()

        # Record in history
        self.task_history.append(
            {
                "task": task,
                "steps": len(steps),
                "success": result.success,
                "timestamp": datetime.now().isoformat(),
            }
        )

        return {
            "success": result.success,
            "steps_executed": result.steps_executed,
            "steps_failed": result.steps_failed,
            "result": result.final_result,
            "created_modules": result.context.created_modules,
            "execution_time_ms": result.execution_time_ms,
        }


# Global agent instance
autonomous_agent = AutonomousAgent()

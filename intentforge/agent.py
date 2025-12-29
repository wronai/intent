"""
IntentForge Autonomous Agent

Lightweight, fully autonomous system that:
- Builds, tests, and reuses modules automatically
- Makes context-aware decisions
- Works like a programmer/office worker
- Minimal core, maximum reusability
"""

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class TaskType(Enum):
    CODE = "code"
    ANALYSIS = "analysis"
    BUILD = "build"
    TEST = "test"
    DEPLOY = "deploy"
    RESEARCH = "research"
    DOCUMENT = "document"


class ModuleStatus(Enum):
    DRAFT = "draft"
    TESTING = "testing"
    READY = "ready"
    DEPRECATED = "deprecated"


@dataclass
class Module:
    """Reusable module built by the agent"""

    id: str
    name: str
    description: str
    code: str
    version: str = "1.0.0"
    status: ModuleStatus = ModuleStatus.DRAFT
    tests_passed: int = 0
    tests_total: int = 0
    dependencies: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: str = ""
    use_count: int = 0
    tags: list[str] = field(default_factory=list)


@dataclass
class Context:
    """Context for decision making"""

    task: str
    history: list[dict] = field(default_factory=list)
    available_modules: list[str] = field(default_factory=list)
    current_code: str = ""
    errors: list[str] = field(default_factory=list)
    environment: dict = field(default_factory=dict)


@dataclass
class Decision:
    """Agent decision"""

    action: str
    params: dict = field(default_factory=dict)
    reasoning: str = ""
    confidence: float = 0.0
    alternatives: list[str] = field(default_factory=list)


@dataclass
class TaskResult:
    """Result of task execution"""

    success: bool
    output: Any = None
    module_created: Module | None = None
    modules_used: list[str] = field(default_factory=list)
    decisions_made: list[Decision] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)


class ModuleRegistry:
    """
    Registry for reusable modules.
    Stores modules in SQLite for persistence.
    """

    def __init__(self, db_path: str = "modules.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS modules (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                code TEXT NOT NULL,
                version TEXT DEFAULT '1.0.0',
                status TEXT DEFAULT 'draft',
                tests_passed INTEGER DEFAULT 0,
                tests_total INTEGER DEFAULT 0,
                dependencies TEXT DEFAULT '[]',
                created_at TEXT,
                last_used TEXT,
                use_count INTEGER DEFAULT 0,
                tags TEXT DEFAULT '[]'
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_modules_name ON modules(name)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_modules_tags ON modules(tags)
        """)
        conn.commit()
        conn.close()

    def register(self, module: Module) -> bool:
        """Register a new module"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO modules
                (id, name, description, code, version, status, tests_passed,
                 tests_total, dependencies, created_at, last_used, use_count, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    module.id,
                    module.name,
                    module.description,
                    module.code,
                    module.version,
                    module.status.value,
                    module.tests_passed,
                    module.tests_total,
                    json.dumps(module.dependencies),
                    module.created_at,
                    module.last_used,
                    module.use_count,
                    json.dumps(module.tags),
                ),
            )
            conn.commit()
            logger.info(f"Registered module: {module.name} v{module.version}")
            return True
        except Exception as e:
            logger.error(f"Failed to register module: {e}")
            return False
        finally:
            conn.close()

    def get(self, module_id: str) -> Module | None:
        """Get module by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT * FROM modules WHERE id = ?", (module_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return self._row_to_module(row)
        return None

    def find(self, query: str, tags: list[str] | None = None) -> list[Module]:
        """Find modules by name/description or tags"""
        conn = sqlite3.connect(self.db_path)

        if tags:
            # Search by tags
            cursor = conn.execute(
                f"""
                SELECT * FROM modules
                WHERE ({" OR ".join(["tags LIKE ?" for _ in tags])})
                ORDER BY use_count DESC
                LIMIT 10
            """,
                [f"%{tag}%" for tag in tags],
            )
        else:
            # Search by name/description
            cursor = conn.execute(
                """
                SELECT * FROM modules
                WHERE (name LIKE ? OR description LIKE ?)
                ORDER BY use_count DESC
                LIMIT 10
            """,
                (f"%{query}%", f"%{query}%"),
            )

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_module(row) for row in rows]

    def use(self, module_id: str) -> Module | None:
        """Get and mark module as used"""
        module = self.get(module_id)
        if module:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """
                UPDATE modules
                SET use_count = use_count + 1, last_used = ?
                WHERE id = ?
            """,
                (datetime.now().isoformat(), module_id),
            )
            conn.commit()
            conn.close()
            module.use_count += 1
        return module

    def list_all(self) -> list[Module]:
        """List all modules"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT * FROM modules ORDER BY use_count DESC")
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_module(row) for row in rows]

    def _row_to_module(self, row) -> Module:
        """Convert database row to Module"""
        return Module(
            id=row[0],
            name=row[1],
            description=row[2],
            code=row[3],
            version=row[4],
            status=ModuleStatus(row[5]),
            tests_passed=row[6],
            tests_total=row[7],
            dependencies=json.loads(row[8]),
            created_at=row[9],
            last_used=row[10] or "",
            use_count=row[11],
            tags=json.loads(row[12]),
        )


class ContextManager:
    """
    Manages context for decision making.
    Fetches relevant information from various sources.
    """

    def __init__(self, registry: ModuleRegistry):
        self.registry = registry
        self.history: list[dict] = []

    async def build_context(self, task: str, code: str = "") -> Context:
        """Build context for a task"""
        ctx = Context(task=task, current_code=code)

        # Add conversation history (last 10)
        ctx.history = self.history[-10:]

        # Find relevant modules
        keywords = self._extract_keywords(task)
        relevant_modules = []
        for keyword in keywords:
            modules = self.registry.find(keyword)
            relevant_modules.extend([m.name for m in modules])
        ctx.available_modules = list(set(relevant_modules))[:5]

        # Environment info
        ctx.environment = {
            "python_version": "3.11",
            "has_docker": True,
            "has_git": True,
        }

        return ctx

    def add_to_history(self, entry: dict):
        """Add entry to history"""
        entry["timestamp"] = datetime.now().isoformat()
        self.history.append(entry)
        # Keep only last 100 entries
        self.history = self.history[-100:]

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text for module search"""
        # Simple keyword extraction
        stopwords = {"a", "the", "is", "in", "to", "for", "of", "and", "or", "with"}
        words = text.lower().split()
        keywords = [w for w in words if len(w) > 3 and w not in stopwords]
        return keywords[:5]


class DecisionEngine:
    """
    Makes autonomous decisions based on context.
    Uses LLM for complex decisions, rules for simple ones.
    """

    def __init__(self):
        self.rules: list[tuple[Callable[[Context], bool], str, dict]] = []
        self._register_default_rules()

    def _register_default_rules(self):
        """Register default decision rules"""
        # Rule: If module exists, reuse it
        self.rules.append(
            (
                lambda ctx: len(ctx.available_modules) > 0,
                "reuse_module",
                {"reason": "Existing module found"},
            )
        )

        # Rule: If error contains "import", might need dependency
        self.rules.append(
            (
                lambda ctx: any("import" in e.lower() for e in ctx.errors),
                "install_dependency",
                {"reason": "Import error detected"},
            )
        )

        # Rule: If code has syntax error, fix it
        self.rules.append(
            (
                lambda ctx: any("syntax" in e.lower() for e in ctx.errors),
                "fix_syntax",
                {"reason": "Syntax error detected"},
            )
        )

    async def decide(self, context: Context) -> Decision:
        """Make a decision based on context"""
        # Try rules first (fast path)
        for condition, action, params in self.rules:
            if condition(context):
                return Decision(
                    action=action,
                    params=params,
                    reasoning=params.get("reason", "Rule matched"),
                    confidence=0.9,
                )

        # Use LLM for complex decisions
        return await self._llm_decide(context)

    async def _llm_decide(self, context: Context) -> Decision:
        """Use LLM for complex decision making"""
        try:
            from .services import ChatService

            chat = ChatService()

            prompt = f"""You are an autonomous programming agent. Based on the context, decide what action to take.

TASK: {context.task}

AVAILABLE MODULES: {", ".join(context.available_modules) or "None"}

CURRENT CODE:
{context.current_code[:500] if context.current_code else "None"}

ERRORS: {", ".join(context.errors) or "None"}

Choose ONE action and respond in JSON:
{{
    "action": "generate_code|reuse_module|fix_error|build_module|test_module|research",
    "params": {{"module_name": "...", "fix_type": "...", etc}},
    "reasoning": "Brief explanation"
}}"""

            response = await chat.send(
                message=prompt,
                system="You are a decision engine. Respond only with valid JSON.",
            )

            if response.get("success"):
                import re

                text = response.get("response", "{}")
                # Extract JSON from response
                json_match = re.search(r"\{[\s\S]*\}", text)
                if json_match:
                    data = json.loads(json_match.group())
                    return Decision(
                        action=data.get("action", "generate_code"),
                        params=data.get("params", {}),
                        reasoning=data.get("reasoning", ""),
                        confidence=0.7,
                    )
        except Exception as e:
            logger.warning(f"LLM decision failed: {e}")

        # Default action
        return Decision(
            action="generate_code",
            params={},
            reasoning="Default action - no specific rule matched",
            confidence=0.5,
        )


class ModuleBuilder:
    """
    Automatically builds, tests, and registers modules.
    """

    def __init__(self, registry: ModuleRegistry):
        self.registry = registry

    async def build_from_code(
        self, code: str, name: str, description: str, tags: list[str] | None = None
    ) -> Module:
        """Build a module from code"""
        module_id = hashlib.md5(f"{name}:{code}".encode()).hexdigest()[:12]

        module = Module(
            id=module_id,
            name=name,
            description=description,
            code=code,
            tags=tags or [],
        )

        # Test the module
        module = await self.test_module(module)

        # Always register (draft or ready)
        if module.tests_passed > 0 and module.tests_passed == module.tests_total:
            module.status = ModuleStatus.READY
        self.registry.register(module)

        return module

    async def test_module(self, module: Module) -> Module:
        """Test a module automatically"""
        from .code_tester import CodeTester

        tester = CodeTester()

        # Generate and run tests
        try:
            test_result = await tester.generate_and_run_tests(
                code=module.code, intent=module.description
            )

            module.tests_total = test_result.get("total", 0)
            module.tests_passed = test_result.get("passed", 0)
            module.status = ModuleStatus.TESTING

            if module.tests_passed == module.tests_total and module.tests_total > 0:
                module.status = ModuleStatus.READY

        except Exception as e:
            logger.warning(f"Module testing failed: {e}")
            module.tests_total = 1
            module.tests_passed = 0

        return module

    async def improve_module(self, module: Module, feedback: str) -> Module:
        """Improve a module based on feedback"""
        from .services import ChatService

        chat = ChatService()

        prompt = f"""Improve this module based on feedback.

MODULE: {module.name}
DESCRIPTION: {module.description}

CURRENT CODE:
```python
{module.code}
```

FEEDBACK: {feedback}

Return ONLY the improved Python code, no explanations."""

        response = await chat.send(message=prompt, system="You are a code improvement expert.")

        if response.get("success"):
            import re

            text = response.get("response", "")
            code_match = re.search(r"```python\n([\s\S]*?)```", text)
            if code_match:
                new_code = code_match.group(1)
                # Create new version
                old_version = module.version.split(".")
                new_version = f"{old_version[0]}.{int(old_version[1]) + 1}.0"

                new_module = Module(
                    id=module.id + "_v" + new_version.replace(".", ""),
                    name=module.name,
                    description=module.description,
                    code=new_code,
                    version=new_version,
                    tags=module.tags,
                )

                return await self.test_module(new_module)

        return module


class AutonomousAgent:
    """
    Fully autonomous agent that builds, tests, and reuses modules.
    Works like a programmer - makes decisions based on context.
    """

    def __init__(self, workspace: str = "/tmp/agent_workspace"):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

        db_path = str(self.workspace / "modules.db")
        self.registry = ModuleRegistry(db_path)
        self.context_mgr = ContextManager(self.registry)
        self.decision_engine = DecisionEngine()
        self.module_builder = ModuleBuilder(self.registry)

    async def execute(self, task: str, code: str = "") -> TaskResult:
        """
        Execute a task autonomously.

        1. Build context
        2. Make decision
        3. Execute action
        4. Build/reuse modules as needed
        5. Return result
        """
        result = TaskResult(success=False, logs=[])
        result.logs.append(f"📋 Task: {task}")

        # Build context
        context = await self.context_mgr.build_context(task, code)
        result.logs.append(f"📊 Context: {len(context.available_modules)} modules available")

        # Decision loop (max 5 iterations)
        for iteration in range(5):
            result.logs.append(f"\n🔄 Iteration {iteration + 1}")

            # Make decision
            decision = await self.decision_engine.decide(context)
            result.decisions_made.append(decision)
            result.logs.append(f"🧠 Decision: {decision.action} ({decision.reasoning})")

            # Execute action
            action_result = await self._execute_action(decision, context)

            if action_result.get("success"):
                result.success = True
                result.output = action_result.get("output")

                # Build module if code was generated
                if decision.action == "generate_code" and action_result.get("code"):
                    module = await self.module_builder.build_from_code(
                        code=action_result["code"],
                        name=self._generate_module_name(task),
                        description=task,
                        tags=self.context_mgr._extract_keywords(task),
                    )
                    result.module_created = module
                    result.logs.append(
                        f"📦 Module created: {module.name} v{module.version} "
                        f"({module.tests_passed}/{module.tests_total} tests)"
                    )

                break

            # Update context with errors
            if action_result.get("error"):
                context.errors.append(action_result["error"])
                result.logs.append(f"❌ Error: {action_result['error'][:100]}")

        # Record in history
        self.context_mgr.add_to_history(
            {"task": task, "success": result.success, "modules_used": result.modules_used}
        )

        return result

    async def _execute_action(self, decision: Decision, context: Context) -> dict:
        """Execute a decision action"""
        action = decision.action
        params = decision.params

        if action == "reuse_module":
            # Find and use existing module
            module_name = params.get("module_name") or (
                context.available_modules[0] if context.available_modules else None
            )
            if module_name:
                modules = self.registry.find(module_name)
                if modules:
                    module = self.registry.use(modules[0].id)
                    return {"success": True, "output": module.code, "module": module}
            return {"success": False, "error": "Module not found"}

        elif action == "generate_code":
            # Generate new code
            from .services import ChatService

            chat = ChatService()
            response = await chat.send(
                message=f"Write Python code for: {context.task}",
                system="You are a Python expert. Write clean, working code. Return only code.",
            )

            if response.get("success"):
                import re

                text = response.get("response", "")
                code_match = re.search(r"```python\n([\s\S]*?)```", text)
                code = code_match.group(1) if code_match else text
                return {"success": True, "code": code, "output": code}

            return {"success": False, "error": response.get("error", "Generation failed")}

        elif action == "fix_error":
            # Fix error in code
            from .code_runner import CodeRunner

            runner = CodeRunner()
            result = await runner.run(context.current_code, auto_fix=True)
            return {
                "success": result.success,
                "code": result.final_code,
                "output": result.stdout,
                "error": result.stderr if not result.success else None,
            }

        elif action == "build_module":
            # Build a module from current code
            if context.current_code:
                module = await self.module_builder.build_from_code(
                    code=context.current_code,
                    name=params.get("name", "unnamed_module"),
                    description=context.task,
                )
                return {"success": module.status == ModuleStatus.READY, "module": module}
            return {"success": False, "error": "No code to build"}

        elif action == "test_module":
            # Test existing module
            module_id = params.get("module_id")
            if module_id:
                module = self.registry.get(module_id)
                if module:
                    module = await self.module_builder.test_module(module)
                    self.registry.register(module)
                    return {"success": module.tests_passed == module.tests_total}
            return {"success": False, "error": "Module not found"}

        else:
            return {"success": False, "error": f"Unknown action: {action}"}

    def _generate_module_name(self, task: str) -> str:
        """Generate module name from task"""
        words = task.lower().split()[:3]
        return "_".join(w for w in words if w.isalnum())[:30] or "module"

    def list_modules(self) -> list[Module]:
        """List all available modules"""
        return self.registry.list_all()


# Convenience function
async def run_agent(task: str, code: str = "") -> dict:
    """Run autonomous agent on a task"""
    agent = AutonomousAgent()
    result = await agent.execute(task, code)

    return {
        "success": result.success,
        "output": result.output,
        "module_created": (
            {
                "name": result.module_created.name,
                "version": result.module_created.version,
                "status": result.module_created.status.value,
                "tests": f"{result.module_created.tests_passed}/{result.module_created.tests_total}",
            }
            if result.module_created
            else None
        ),
        "modules_used": result.modules_used,
        "decisions": [
            {"action": d.action, "reasoning": d.reasoning} for d in result.decisions_made
        ],
        "logs": result.logs,
    }

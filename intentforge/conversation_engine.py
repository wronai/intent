"""
IntentForge Conversation Engine

Multi-threaded, branching conversation system for autonomous error resolution.

Architecture:
- ConversationThread: Single conversation context with history
- ConversationBrancher: Spawns sub-conversations for each detected problem
- ThreadManager: Coordinates parallel conversations
- LLMAnalyzer: Replaces hardcoded logic with LLM-driven analysis

Flow:
1. Main conversation detects error
2. Brancher spawns sub-conversation for the error
3. Sub-conversation uses LLM to analyze and fix
4. Results merge back to main thread
5. Continue until all issues resolved
"""

import asyncio
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ConversationStatus(Enum):
    """Status of a conversation thread"""

    ACTIVE = "active"
    WAITING = "waiting"
    RESOLVED = "resolved"
    FAILED = "failed"
    MERGED = "merged"


class ProblemType(Enum):
    """Types of problems that spawn sub-conversations"""

    CODE_ERROR = "code_error"
    MISSING_FILE = "missing_file"
    MISSING_DEPENDENCY = "missing_dependency"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    LOGIC_ERROR = "logic_error"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class Message:
    """Single message in conversation"""

    role: str  # system, user, assistant, error, fix
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)


@dataclass
class Problem:
    """Detected problem that needs resolution"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: ProblemType = ProblemType.UNKNOWN
    description: str = ""
    context: str = ""
    source_code: str = ""
    error_output: str = ""
    priority: int = 1
    resolved: bool = False
    resolution: str = ""


@dataclass
class ConversationThread:
    """Single conversation thread with history"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: str = None
    status: ConversationStatus = ConversationStatus.ACTIVE
    messages: list = field(default_factory=list)
    problems: list = field(default_factory=list)
    context: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    resolved_problems: list = field(default_factory=list)
    child_threads: list = field(default_factory=list)


class LLMAnalyzer:
    """
    LLM-driven analysis replacing hardcoded logic.

    Instead of regex patterns, uses LLM to:
    - Classify error types
    - Suggest fixes
    - Generate corrected code
    - Determine if problem is resolved
    """

    def __init__(self):
        self._chat_service = None

    @property
    def chat(self):
        if self._chat_service is None:
            from .services import ChatService

            self._chat_service = ChatService()
        return self._chat_service

    async def analyze_error(self, error_output: str, code: str = "", context: str = "") -> dict:
        """
        Use LLM to analyze error and classify it.
        Replaces hardcoded regex patterns.
        """
        prompt = f"""Analyze this error and respond with JSON only:

ERROR OUTPUT:
{error_output[:2000]}

CODE (if available):
{code[:1500] if code else "Not provided"}

CONTEXT:
{context[:500] if context else "Not provided"}

Respond with JSON:
{{
    "error_type": "missing_dependency|syntax_error|runtime_error|missing_file|logic_error|configuration|unknown",
    "root_cause": "Brief description of root cause",
    "affected_component": "What part of code/system is affected",
    "suggested_fix": "How to fix this",
    "can_auto_fix": true/false,
    "fix_code": "Corrected code if applicable",
    "confidence": 0.0-1.0
}}
"""
        response = await self.chat.send(
            message=prompt,
            system="You are an error analysis expert. Return only valid JSON, no explanations.",
        )

        if not response.get("success"):
            return {"error_type": "unknown", "can_auto_fix": False}

        try:
            import json
            import re

            text = response.get("response", "{}")
            # Extract JSON from response
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(text)
        except Exception:
            return {"error_type": "unknown", "can_auto_fix": False}

    async def generate_fix(self, problem: Problem) -> dict:
        """
        Use LLM to generate fix for a problem.
        """
        prompt = f"""Fix this problem:

PROBLEM TYPE: {problem.type.value}
DESCRIPTION: {problem.description}

ERROR OUTPUT:
{problem.error_output[:1500]}

ORIGINAL CODE:
{problem.source_code[:2000] if problem.source_code else "Not provided"}

Generate the fixed code. Return JSON:
{{
    "fixed_code": "Complete fixed code here",
    "explanation": "What was changed and why",
    "additional_steps": ["Any additional steps needed"],
    "dependencies": ["Any new dependencies needed"]
}}
"""
        response = await self.chat.send(
            message=prompt,
            system="You are a code repair expert. Return only valid JSON with fixed code.",
        )

        if not response.get("success"):
            return {"fixed_code": None, "explanation": "LLM request failed"}

        try:
            import json
            import re

            text = response.get("response", "{}")
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(text)
        except Exception:
            return {"fixed_code": None, "explanation": "Failed to parse LLM response"}

    async def should_branch(self, problems: list) -> list:
        """
        Determine which problems need separate conversation threads.
        """
        if len(problems) <= 1:
            return problems

        prompt = f"""Given these problems, determine which should be solved in parallel vs sequentially:

PROBLEMS:
{chr(10).join([f"- {p.type.value}: {p.description}" for p in problems])}

Return JSON:
{{
    "parallel_groups": [
        {{"problems": [0, 1], "reason": "Independent issues"}},
        {{"problems": [2], "reason": "Depends on first group"}}
    ],
    "recommended_order": [0, 1, 2]
}}
"""
        await self.chat.send(
            message=prompt,
            system="You are a problem prioritization expert. Return only valid JSON.",
        )

        # Default: all problems can be handled in parallel
        return problems


class ConversationBrancher:
    """
    Spawns sub-conversations for each detected problem.
    Manages the branching tree of conversations.
    """

    def __init__(self):
        self.threads: dict[str, ConversationThread] = {}
        self.analyzer = LLMAnalyzer()
        self.on_problem_detected: Callable = None
        self.on_problem_resolved: Callable = None

    def create_thread(
        self, parent_id: str | None = None, context: dict | None = None
    ) -> ConversationThread:
        """Create new conversation thread"""
        thread = ConversationThread(
            parent_id=parent_id,
            context=context or {},
        )
        self.threads[thread.id] = thread

        if parent_id and parent_id in self.threads:
            self.threads[parent_id].child_threads.append(thread.id)

        logger.info(f"Created thread {thread.id} (parent: {parent_id})")
        return thread

    async def detect_problems(
        self,
        thread_id: str,
        output: str,
        code: str = "",
        context: str = "",
    ) -> list[Problem]:
        """
        Detect problems in output using LLM analysis.
        """
        if thread_id not in self.threads:
            return []

        thread = self.threads[thread_id]

        # Use LLM to analyze errors
        analysis = await self.analyzer.analyze_error(output, code, context)

        problems = []
        if analysis.get("error_type") != "unknown" or "error" in output.lower():
            problem = Problem(
                type=ProblemType(analysis.get("error_type", "unknown")),
                description=analysis.get("root_cause", output[:200]),
                context=context,
                source_code=code,
                error_output=output,
            )
            problems.append(problem)
            thread.problems.append(problem)

            # Callback for UI updates
            if self.on_problem_detected:
                await self._call_callback(self.on_problem_detected, thread, problem)

        return problems

    async def branch_for_problem(
        self,
        parent_thread_id: str,
        problem: Problem,
    ) -> ConversationThread:
        """
        Create a sub-conversation thread to solve a specific problem.
        """
        parent = self.threads.get(parent_thread_id)
        if not parent:
            raise ValueError(f"Parent thread {parent_thread_id} not found")

        # Create child thread
        child = self.create_thread(
            parent_id=parent_thread_id,
            context={
                "problem_id": problem.id,
                "problem_type": problem.type.value,
                "inherited_context": parent.context,
            },
        )

        # Add initial messages
        child.messages.append(
            Message(
                role="system",
                content=f"Sub-conversation to resolve: {problem.type.value}",
            )
        )
        child.messages.append(
            Message(
                role="error",
                content=problem.error_output[:1000],
            )
        )

        child.problems.append(problem)

        logger.info(f"Branched thread {child.id} for problem {problem.id}")
        return child

    async def resolve_in_thread(
        self,
        thread_id: str,
        problem: Problem,
        max_attempts: int = 5,
    ) -> dict:
        """
        Resolve a problem within its conversation thread.
        Uses LLM for analysis and fixing at each step.
        """
        thread = self.threads.get(thread_id)
        if not thread:
            return {"resolved": False, "reason": "Thread not found"}

        current_error = problem.error_output

        for attempt in range(max_attempts):
            thread.messages.append(
                Message(
                    role="system",
                    content=f"Attempt {attempt + 1}/{max_attempts} to resolve {problem.type.value}",
                )
            )

            # Get fix from LLM
            fix_result = await self.analyzer.generate_fix(problem)

            if not fix_result.get("fixed_code"):
                thread.messages.append(
                    Message(
                        role="assistant",
                        content=f"Could not generate fix: {fix_result.get('explanation', 'Unknown')}",
                    )
                )
                continue

            fixed_code = fix_result["fixed_code"]
            thread.messages.append(
                Message(
                    role="fix",
                    content=fixed_code,
                    metadata={"explanation": fix_result.get("explanation", "")},
                )
            )

            # Execute fixed code
            from .code_runner import run_with_autofix

            exec_result = await run_with_autofix(
                code=fixed_code,
                language="python",
                max_retries=1,
                auto_install=True,
            )

            if exec_result.get("success"):
                problem.resolved = True
                problem.resolution = fixed_code
                thread.resolved_problems.append(problem)
                thread.status = ConversationStatus.RESOLVED

                thread.messages.append(
                    Message(
                        role="assistant",
                        content=f"✅ Problem resolved on attempt {attempt + 1}",
                        metadata={"output": exec_result.get("output", "")},
                    )
                )

                if self.on_problem_resolved:
                    await self._call_callback(self.on_problem_resolved, thread, problem)

                return {
                    "resolved": True,
                    "attempts": attempt + 1,
                    "fixed_code": fixed_code,
                    "output": exec_result.get("output", ""),
                }

            # Update problem with new error for next iteration
            problem.error_output = exec_result.get("error", "")
            current_error = problem.error_output

            thread.messages.append(
                Message(
                    role="error",
                    content=f"Attempt {attempt + 1} failed: {current_error[:500]}",
                )
            )

        thread.status = ConversationStatus.FAILED
        return {
            "resolved": False,
            "attempts": max_attempts,
            "last_error": current_error,
        }

    async def merge_results(self, parent_thread_id: str) -> dict:
        """
        Merge results from child threads back to parent.
        """
        parent = self.threads.get(parent_thread_id)
        if not parent:
            return {"success": False, "reason": "Parent thread not found"}

        results = {
            "total_children": len(parent.child_threads),
            "resolved": 0,
            "failed": 0,
            "merged_fixes": [],
        }

        for child_id in parent.child_threads:
            child = self.threads.get(child_id)
            if not child:
                continue

            if child.status == ConversationStatus.RESOLVED:
                results["resolved"] += 1
                for problem in child.resolved_problems:
                    results["merged_fixes"].append(
                        {
                            "problem_id": problem.id,
                            "type": problem.type.value,
                            "resolution": problem.resolution,
                        }
                    )
                child.status = ConversationStatus.MERGED
            else:
                results["failed"] += 1

        parent.messages.append(
            Message(
                role="system",
                content=f"Merged {results['resolved']}/{results['total_children']} child threads",
                metadata=results,
            )
        )

        return results

    async def _call_callback(self, callback: Callable, *args):
        """Call callback function"""
        if asyncio.iscoroutinefunction(callback):
            await callback(*args)
        else:
            callback(*args)


class ThreadManager:
    """
    Manages parallel execution of conversation threads.
    """

    def __init__(self, max_parallel: int = 5):
        self.max_parallel = max_parallel
        self.brancher = ConversationBrancher()
        self.active_tasks: dict[str, asyncio.Task] = {}

    async def process_with_branching(
        self,
        code: str,
        intent: str = "",
        auto_branch: bool = True,
    ) -> dict:
        """
        Process code with automatic branching for errors.

        Flow:
        1. Execute code
        2. If errors, detect problems
        3. Branch conversation for each problem
        4. Resolve in parallel
        5. Merge results
        """
        # Create main thread
        main_thread = self.brancher.create_thread(
            context={
                "original_code": code,
                "intent": intent,
            }
        )

        # Initial execution
        from .code_runner import run_with_autofix

        result = await run_with_autofix(code, language="python", max_retries=1)

        if result.get("success"):
            main_thread.status = ConversationStatus.RESOLVED
            return {
                "success": True,
                "output": result.get("output", ""),
                "threads": 1,
                "branches": 0,
            }

        # Detect problems
        problems = await self.brancher.detect_problems(
            thread_id=main_thread.id,
            output=result.get("error", ""),
            code=code,
            context=intent,
        )

        if not problems or not auto_branch:
            return {
                "success": False,
                "error": result.get("error", ""),
                "problems_detected": len(problems),
            }

        # Branch for each problem
        child_threads = []
        for problem in problems:
            child = await self.brancher.branch_for_problem(main_thread.id, problem)
            child_threads.append((child, problem))

        # Resolve in parallel
        tasks = []
        for child, problem in child_threads:
            task = asyncio.create_task(self.brancher.resolve_in_thread(child.id, problem))
            tasks.append(task)
            self.active_tasks[child.id] = task

        # Wait for all
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results
        merge_result = await self.brancher.merge_results(main_thread.id)

        # Clean up tasks
        for child, _ in child_threads:
            self.active_tasks.pop(child.id, None)

        # Collect final result
        all_resolved = all(
            isinstance(r, dict) and r.get("resolved")
            for r in results
            if not isinstance(r, Exception)
        )

        final_code = code
        for r in results:
            if isinstance(r, dict) and r.get("fixed_code"):
                final_code = r["fixed_code"]

        return {
            "success": all_resolved,
            "final_code": final_code,
            "threads": 1 + len(child_threads),
            "problems_resolved": merge_result.get("resolved", 0),
            "problems_failed": merge_result.get("failed", 0),
            "conversation_tree": self._build_tree(main_thread.id),
        }

    def _build_tree(self, thread_id: str, depth: int = 0) -> dict:
        """Build conversation tree structure"""
        thread = self.brancher.threads.get(thread_id)
        if not thread:
            return {}

        return {
            "id": thread.id,
            "status": thread.status.value,
            "messages": len(thread.messages),
            "problems": len(thread.problems),
            "resolved": len(thread.resolved_problems),
            "children": [
                self._build_tree(child_id, depth + 1) for child_id in thread.child_threads
            ],
        }

    def get_thread_status(self, thread_id: str) -> dict:
        """Get status of a specific thread"""
        thread = self.brancher.threads.get(thread_id)
        if not thread:
            return {"error": "Thread not found"}

        return {
            "id": thread.id,
            "status": thread.status.value,
            "messages": [
                {"role": m.role, "content": m.content[:200], "timestamp": m.timestamp}
                for m in thread.messages[-10:]
            ],
            "problems": len(thread.problems),
            "resolved": len(thread.resolved_problems),
        }


# Global instances
thread_manager = ThreadManager()


async def process_with_auto_conversation(
    code: str,
    intent: str = "",
    auto_branch: bool = True,
) -> dict:
    """
    Convenience function for processing code with automatic conversation branching.
    """
    return await thread_manager.process_with_branching(
        code=code,
        intent=intent,
        auto_branch=auto_branch,
    )

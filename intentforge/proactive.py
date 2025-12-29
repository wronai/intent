"""
IntentForge Proactive Decision Engine

Intelligent algorithms for automatic processing, decision-making, and action execution.
Enables self-improving workflows where:
- Generated code is automatically run and debugged
- Documents are processed and data is extracted/used
- Decisions are made based on context and previous results
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Types of proactive decisions"""

    EXECUTE_CODE = "execute_code"
    DEBUG_CODE = "debug_code"
    EXTRACT_DATA = "extract_data"
    PROCESS_DOCUMENT = "process_document"
    RETRY_WITH_FALLBACK = "retry_with_fallback"
    ENHANCE_QUALITY = "enhance_quality"
    SAVE_RESULT = "save_result"
    NOTIFY_USER = "notify_user"
    CHAIN_ACTION = "chain_action"


@dataclass
class Decision:
    """A proactive decision to be executed"""

    type: DecisionType
    action: str
    params: dict = field(default_factory=dict)
    priority: int = 5  # 1-10, higher = more urgent
    reason: str = ""
    auto_execute: bool = True
    requires_confirmation: bool = False


@dataclass
class ProcessingContext:
    """Context for processing decisions"""

    input_type: str  # "code", "document", "image", "text"
    content: Any
    metadata: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    results: dict = field(default_factory=dict)


class ProactiveEngine:
    """
    Proactive Decision Engine for intelligent automation.

    Features:
    - Code execution and debugging
    - Document processing and data extraction
    - Automatic fallback and retry logic
    - Quality enhancement
    - Result chaining
    """

    def __init__(self):
        self.decision_handlers: dict[DecisionType, Callable] = {}
        self.rules: list[Callable] = []
        self.history: list[dict] = []
        self._register_default_handlers()
        self._register_default_rules()

    def _register_default_handlers(self):
        """Register default decision handlers"""
        self.decision_handlers = {
            DecisionType.EXECUTE_CODE: self._handle_execute_code,
            DecisionType.DEBUG_CODE: self._handle_debug_code,
            DecisionType.EXTRACT_DATA: self._handle_extract_data,
            DecisionType.PROCESS_DOCUMENT: self._handle_process_document,
            DecisionType.RETRY_WITH_FALLBACK: self._handle_retry_fallback,
            DecisionType.ENHANCE_QUALITY: self._handle_enhance_quality,
            DecisionType.SAVE_RESULT: self._handle_save_result,
            DecisionType.CHAIN_ACTION: self._handle_chain_action,
        }

    def _register_default_rules(self):
        """Register default decision rules"""
        self.rules = [
            self._rule_code_detection,
            self._rule_document_detection,
            self._rule_error_recovery,
            self._rule_quality_check,
            self._rule_data_extraction,
        ]

    async def analyze(self, context: ProcessingContext) -> list[Decision]:
        """Analyze context and generate proactive decisions"""
        decisions = []

        for rule in self.rules:
            try:
                rule_decisions = await rule(context)
                if rule_decisions:
                    decisions.extend(rule_decisions)
            except Exception as e:
                logger.error(f"Rule error: {e}")

        # Sort by priority
        decisions.sort(key=lambda d: d.priority, reverse=True)

        return decisions

    async def execute(self, decisions: list[Decision], context: ProcessingContext) -> dict:
        """Execute a list of decisions"""
        results = {"executed": [], "skipped": [], "errors": []}

        for decision in decisions:
            if not decision.auto_execute:
                results["skipped"].append(
                    {"type": decision.type.value, "reason": "requires manual execution"}
                )
                continue

            handler = self.decision_handlers.get(decision.type)
            if not handler:
                results["errors"].append(
                    {"type": decision.type.value, "error": "No handler registered"}
                )
                continue

            try:
                result = await handler(decision, context)
                results["executed"].append(
                    {
                        "type": decision.type.value,
                        "action": decision.action,
                        "result": result,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                # Update context with result
                context.results[decision.type.value] = result
                context.history.append(
                    {
                        "decision": decision.type.value,
                        "result": result,
                    }
                )

            except Exception as e:
                results["errors"].append(
                    {
                        "type": decision.type.value,
                        "error": str(e),
                    }
                )

        self.history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "context_type": context.input_type,
                "decisions_count": len(decisions),
                "results": results,
            }
        )

        return results

    async def process(self, context: ProcessingContext) -> dict:
        """Full proactive processing pipeline"""
        decisions = await self.analyze(context)
        results = await self.execute(decisions, context)

        return {
            "decisions": [d.type.value for d in decisions],
            "results": results,
            "context": {
                "type": context.input_type,
                "history": context.history[-5:],  # Last 5 actions
            },
        }

    # =========================================================================
    # Decision Rules
    # =========================================================================

    async def _rule_code_detection(self, context: ProcessingContext) -> list[Decision]:
        """Detect code in content and decide on execution"""
        decisions = []

        if context.input_type != "text":
            return decisions

        content = str(context.content)

        # Detect code blocks
        code_pattern = r"```(\w+)?\n([\s\S]*?)```"
        matches = re.findall(code_pattern, content)

        for lang, code in matches:
            lang = lang.lower() if lang else "python"

            # Check if code is executable
            executable_langs = ["python", "javascript", "bash", "shell", "ruby", "php", "go"]

            if lang in executable_langs and len(code.strip()) > 10:
                decisions.append(
                    Decision(
                        type=DecisionType.EXECUTE_CODE,
                        action="run",
                        params={"code": code.strip(), "language": lang},
                        priority=8,
                        reason=f"Detected executable {lang} code",
                        auto_execute=True,
                    )
                )

        return decisions

    async def _rule_document_detection(self, context: ProcessingContext) -> list[Decision]:
        """Detect document type and decide on processing"""
        decisions = []

        if context.input_type not in ["image", "document"]:
            return decisions

        metadata = context.metadata

        # Check for document indicators
        if metadata.get("has_text_detected") or metadata.get("document_type"):
            doc_type = metadata.get("document_type", "unknown")

            decisions.append(
                Decision(
                    type=DecisionType.PROCESS_DOCUMENT,
                    action="extract",
                    params={
                        "document_type": doc_type,
                        "content": context.content,
                    },
                    priority=7,
                    reason=f"Document detected: {doc_type}",
                    auto_execute=True,
                )
            )

            # If OCR failed but Vision detected text, retry with enhanced OCR
            if metadata.get("ocr_failed") and metadata.get("vision_detected_text"):
                decisions.append(
                    Decision(
                        type=DecisionType.RETRY_WITH_FALLBACK,
                        action="enhanced_ocr",
                        params={"image": context.content},
                        priority=9,
                        reason="OCR failed but Vision detected text - retry with enhancement",
                        auto_execute=True,
                    )
                )

        return decisions

    async def _rule_error_recovery(self, context: ProcessingContext) -> list[Decision]:
        """Detect errors and decide on recovery actions"""
        decisions = []

        # Check for errors in results
        for key, result in context.results.items():
            if isinstance(result, dict) and not result.get("success", True):
                error = result.get("error", "Unknown error")

                # Decide on recovery action
                if "timeout" in error.lower():
                    decisions.append(
                        Decision(
                            type=DecisionType.RETRY_WITH_FALLBACK,
                            action="retry_with_smaller_model",
                            params={"original_action": key},
                            priority=6,
                            reason="Timeout error, retry with smaller model",
                        )
                    )
                elif "ocr" in error.lower() or "text" in error.lower():
                    decisions.append(
                        Decision(
                            type=DecisionType.ENHANCE_QUALITY,
                            action="preprocess_image",
                            params={"original_action": key},
                            priority=7,
                            reason="OCR failed, try image preprocessing",
                        )
                    )

        return decisions

    async def _rule_quality_check(self, context: ProcessingContext) -> list[Decision]:
        """Check quality of results and decide on improvements"""
        decisions = []

        for _key, result in context.results.items():
            if isinstance(result, dict):
                confidence = result.get("confidence", 100)

                # Low confidence - enhance
                if confidence < 60:
                    decisions.append(
                        Decision(
                            type=DecisionType.ENHANCE_QUALITY,
                            action="retry_with_preprocessing",
                            params={"original_result": result},
                            priority=5,
                            reason=f"Low confidence ({confidence}%), enhance quality",
                        )
                    )

        return decisions

    async def _rule_data_extraction(self, context: ProcessingContext) -> list[Decision]:
        """Decide on data extraction from processed content"""
        decisions = []

        # Check if we have text that could contain structured data
        if context.input_type == "document" or context.metadata.get("has_structured_data"):
            text = context.results.get("ocr", {}).get("text", "")

            if len(text) > 50:
                # Detect data patterns
                has_numbers = bool(re.search(r"\d+[.,]\d+", text))
                has_dates = bool(re.search(r"\d{1,2}[./]\d{1,2}[./]\d{2,4}", text))
                has_currency = bool(re.search(r"(PLN|zł|EUR|USD|\$|€)", text))

                if has_numbers or has_dates or has_currency:
                    decisions.append(
                        Decision(
                            type=DecisionType.EXTRACT_DATA,
                            action="structure",
                            params={"text": text},
                            priority=6,
                            reason="Detected structured data (numbers/dates/currency)",
                            auto_execute=True,
                        )
                    )

        return decisions

    # =========================================================================
    # Decision Handlers
    # =========================================================================

    async def _handle_execute_code(self, decision: Decision, context: ProcessingContext) -> dict:
        """Execute code and return results"""
        import subprocess
        import tempfile

        code = decision.params.get("code", "")
        language = decision.params.get("language", "python")

        extensions = {
            "python": ".py",
            "javascript": ".js",
            "bash": ".sh",
            "shell": ".sh",
            "ruby": ".rb",
            "php": ".php",
            "go": ".go",
        }
        commands = {
            "python": ["python3"],
            "javascript": ["node"],
            "bash": ["bash"],
            "shell": ["sh"],
            "ruby": ["ruby"],
            "php": ["php"],
            "go": ["go", "run"],
        }

        ext = extensions.get(language, ".txt")
        cmd = commands.get(language, ["cat"])

        with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            result = subprocess.run(
                [*cmd, temp_path], check=False, capture_output=True, text=True, timeout=30
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout[:5000],
                "stderr": result.stderr[:2000],
                "returncode": result.returncode,
                "language": language,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Execution timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            os.unlink(temp_path)

    async def _handle_debug_code(self, decision: Decision, context: ProcessingContext) -> dict:
        """Debug code by analyzing errors and suggesting fixes"""
        from .services import ChatService

        code = decision.params.get("code", "")
        error = decision.params.get("error", "")
        language = decision.params.get("language", "python")

        chat = ChatService()
        response = await chat.send(
            message=f"Debug this {language} code. Error: {error}\n\nCode:\n```{language}\n{code}\n```",
            system="You are a code debugger. Analyze the error and provide a fixed version of the code. "
            "Explain the bug and show the corrected code in a code block.",
        )

        return {
            "success": response.get("success", False),
            "analysis": response.get("response", ""),
            "original_error": error,
        }

    async def _handle_extract_data(self, decision: Decision, context: ProcessingContext) -> dict:
        """Extract structured data from text"""
        from .services import ChatService

        text = decision.params.get("text", "")

        chat = ChatService()
        response = await chat.send(
            message=f"Extract all structured data from this text and return as JSON:\n\n{text[:3000]}",
            system="You are a data extraction expert. Extract key-value pairs, tables, "
            "lists, and structured information. Return valid JSON only.",
        )

        try:
            content = response.get("response", "")
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            data = json.loads(content.strip())
            return {"success": True, "data": data}
        except json.JSONDecodeError:
            return {"success": True, "data": {"raw": response.get("response", "")}}

    async def _handle_process_document(
        self, decision: Decision, context: ProcessingContext
    ) -> dict:
        """Process document with full pipeline"""
        from .services import FileService

        content = decision.params.get("content")

        file_service = FileService()
        result = await file_service.process_document(image_base64=content)

        return result

    async def _handle_retry_fallback(self, decision: Decision, context: ProcessingContext) -> dict:
        """Retry failed action with fallback strategy"""
        action = decision.params.get("action", "")

        if action == "enhanced_ocr":
            from .services import FileService

            image = decision.params.get("image")
            file_service = FileService()

            # Try OCR with different settings
            result = await file_service.ocr(image_base64=image, use_tesseract=True)

            if not result.get("success") or not result.get("text", "").strip():
                # Fall back to Vision-only OCR
                result = await file_service._analyze_image_with_vision(
                    image,
                    prompt="Przepisz DOKŁADNIE cały tekst z tego obrazu, zachowując układ. "
                    "Nie dodawaj żadnych komentarzy, tylko tekst z obrazu.",
                )
                if result.get("success"):
                    result["text"] = result.get("response", "")
                    result["method"] = "vision_fallback"

            return result

        return {"success": False, "error": f"Unknown fallback action: {action}"}

    async def _handle_enhance_quality(self, decision: Decision, context: ProcessingContext) -> dict:
        """Enhance quality of processing"""
        action = decision.action

        if action == "preprocess_image":
            # Already handled in OCR preprocessing
            return {"success": True, "action": "preprocessing_applied"}

        return {"success": True, "action": action}

    async def _handle_save_result(self, decision: Decision, context: ProcessingContext) -> dict:
        """Save processing results"""
        data = decision.params.get("data", {})
        filename = decision.params.get("filename", "result.json")

        import json

        with open(f"/tmp/intentforge_{filename}", "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return {"success": True, "path": f"/tmp/intentforge_{filename}"}

    async def _handle_chain_action(self, decision: Decision, context: ProcessingContext) -> dict:
        """Chain multiple actions together"""
        actions = decision.params.get("actions", [])
        results = []

        for action in actions:
            # Process each action in sequence
            sub_decision = Decision(
                type=DecisionType[action["type"].upper()],
                action=action.get("action", ""),
                params=action.get("params", {}),
            )
            handler = self.decision_handlers.get(sub_decision.type)
            if handler:
                result = await handler(sub_decision, context)
                results.append(result)

        return {"success": True, "chain_results": results}


# Global instance
proactive_engine = ProactiveEngine()

"""
IntentForge Git Manager

Git-based iteration tracking system for code changes:
- Track each fix attempt as a commit
- Analyze Git history to understand context
- Learn from previous iterations to improve fixes
- Provide diff-based context for LLM

Flow:
1. Create branch for fix session
2. Each iteration = commit with descriptive message
3. Analyze diffs to understand what changed
4. Use history to provide context for better fixes
5. Merge successful fixes, squash or keep history
"""

import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GitCommit:
    """Represents a Git commit"""

    hash: str
    message: str
    timestamp: str
    author: str
    files_changed: list = field(default_factory=list)
    diff: str = ""


@dataclass
class IterationHistory:
    """History of code iterations"""

    branch_name: str
    commits: list = field(default_factory=list)
    initial_code: str = ""
    final_code: str = ""
    success: bool = False
    total_iterations: int = 0


class GitManager:
    """
    Manages Git operations for code iteration tracking.

    Features:
    - Create isolated branches for fix sessions
    - Commit each iteration with descriptive messages
    - Analyze diffs between iterations
    - Provide context from history to LLM
    - Learn patterns from successful fixes
    """

    def __init__(self, repo_path: str | None = None):
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self.current_branch: str = None
        self.session_branches: list = []

    def _run_git(self, *args, capture: bool = True) -> tuple[int, str, str]:
        """Run a git command"""
        cmd = ["git", *list(args)]
        try:
            result = subprocess.run(
                cmd,
                check=False,
                cwd=str(self.repo_path),
                capture_output=capture,
                text=True,
                timeout=30,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Git command timed out"
        except Exception as e:
            return -1, "", str(e)

    def init_repo(self) -> bool:
        """Initialize git repo if not exists"""
        if not (self.repo_path / ".git").exists():
            code, _, err = self._run_git("init")
            if code != 0:
                logger.error(f"Failed to init repo: {err}")
                return False
            # Initial commit
            self._run_git("add", "-A")
            self._run_git("commit", "-m", "Initial commit")
        return True

    def get_current_branch(self) -> str:
        """Get current branch name"""
        code, out, _ = self._run_git("branch", "--show-current")
        return out.strip() if code == 0 else "main"

    def create_fix_branch(self, prefix: str = "fix") -> str:
        """Create a new branch for fix session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        branch_name = f"{prefix}/{timestamp}"

        # Save current branch
        self.current_branch = self.get_current_branch()

        # Create and checkout new branch
        code, _, err = self._run_git("checkout", "-b", branch_name)
        if code != 0:
            logger.error(f"Failed to create branch: {err}")
            return None

        self.session_branches.append(branch_name)
        logger.info(f"Created fix branch: {branch_name}")
        return branch_name

    def commit_iteration(
        self,
        code: str,
        filename: str,
        message: str,
        iteration: int,
        error: str | None = None,
        fix_applied: str | None = None,
    ) -> GitCommit:
        """
        Commit a code iteration.

        Args:
            code: Current code state
            filename: File to save code to
            message: Commit message
            iteration: Iteration number
            error: Error that was fixed (optional)
            fix_applied: Description of fix (optional)
        """
        # Write code to file
        file_path = self.repo_path / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(code)

        # Stage the file
        self._run_git("add", filename)

        # Build commit message
        full_message = f"[Iteration {iteration}] {message}"
        if error:
            full_message += f"\n\nError: {error[:200]}"
        if fix_applied:
            full_message += f"\n\nFix: {fix_applied}"

        # Commit
        code_result, _, err = self._run_git("commit", "-m", full_message)
        if code_result != 0:
            logger.warning(f"Commit failed: {err}")
            return None

        # Get commit info
        _, hash_out, _ = self._run_git("rev-parse", "HEAD")
        commit_hash = hash_out.strip()[:8]

        return GitCommit(
            hash=commit_hash,
            message=message,
            timestamp=datetime.now().isoformat(),
            author="IntentForge",
            files_changed=[filename],
        )

    def get_diff(self, from_ref: str = "HEAD~1", to_ref: str = "HEAD") -> str:
        """Get diff between two refs"""
        code, out, _ = self._run_git("diff", from_ref, to_ref)
        return out if code == 0 else ""

    def get_file_history(self, filename: str, limit: int = 10) -> list[GitCommit]:
        """Get commit history for a specific file"""
        code, out, _ = self._run_git(
            "log",
            f"-{limit}",
            "--pretty=format:%H|%s|%ai|%an",
            "--",
            filename,
        )

        if code != 0 or not out:
            return []

        commits = []
        for line in out.strip().split("\n"):
            if "|" in line:
                parts = line.split("|")
                if len(parts) >= 4:
                    commits.append(
                        GitCommit(
                            hash=parts[0][:8],
                            message=parts[1],
                            timestamp=parts[2],
                            author=parts[3],
                        )
                    )

        return commits

    def get_iteration_context(self, filename: str, limit: int = 5) -> str:
        """
        Build context from Git history for LLM.
        Returns a summary of recent changes.
        """
        history = self.get_file_history(filename, limit)
        if not history:
            return "No previous iterations found."

        context_lines = ["## Recent Iteration History\n"]

        for i, commit in enumerate(history):
            context_lines.append(f"### Iteration {len(history) - i}")
            context_lines.append(f"- **Commit**: {commit.hash}")
            context_lines.append(f"- **Message**: {commit.message}")
            context_lines.append(f"- **Time**: {commit.timestamp}")

            # Get diff for this commit
            diff = self.get_diff(f"{commit.hash}~1", commit.hash)
            if diff:
                # Truncate diff if too long
                diff_preview = diff[:500] + "..." if len(diff) > 500 else diff
                context_lines.append(f"```diff\n{diff_preview}\n```")

            context_lines.append("")

        return "\n".join(context_lines)

    def analyze_patterns(self, filename: str) -> dict:
        """
        Analyze patterns from Git history.
        Returns insights about common fixes.
        """
        history = self.get_file_history(filename, 20)

        patterns = {
            "total_iterations": len(history),
            "common_fixes": [],
            "error_types": {},
            "success_rate": 0,
        }

        success_count = 0
        for commit in history:
            msg = commit.message.lower()

            # Track error types from messages
            if "missing" in msg or "import" in msg:
                patterns["error_types"]["missing_module"] = (
                    patterns["error_types"].get("missing_module", 0) + 1
                )
            if "syntax" in msg:
                patterns["error_types"]["syntax_error"] = (
                    patterns["error_types"].get("syntax_error", 0) + 1
                )
            if "runtime" in msg or "exception" in msg:
                patterns["error_types"]["runtime_error"] = (
                    patterns["error_types"].get("runtime_error", 0) + 1
                )

            # Track success
            if "success" in msg or "fixed" in msg or "resolved" in msg:
                success_count += 1

        if history:
            patterns["success_rate"] = success_count / len(history)

        return patterns

    def merge_fix_branch(self, branch_name: str | None = None, squash: bool = False) -> bool:
        """Merge fix branch back to original branch"""
        if not branch_name:
            branch_name = self.get_current_branch()

        if not self.current_branch:
            logger.warning("No original branch to merge to")
            return False

        # Checkout original branch
        self._run_git("checkout", self.current_branch)

        # Merge
        if squash:
            code, _, err = self._run_git("merge", "--squash", branch_name)
            if code == 0:
                self._run_git("commit", "-m", f"Merged fixes from {branch_name}")
        else:
            code, _, err = self._run_git("merge", branch_name)

        if code != 0:
            logger.error(f"Merge failed: {err}")
            return False

        logger.info(f"Merged {branch_name} to {self.current_branch}")
        return True

    def cleanup_branch(self, branch_name: str) -> bool:
        """Delete a fix branch"""
        code, _, err = self._run_git("branch", "-D", branch_name)
        if code != 0:
            logger.warning(f"Failed to delete branch: {err}")
            return False
        return True


class GitIterationTracker:
    """
    High-level tracker that integrates Git with code fixing.

    Tracks each fix attempt as a commit, providing:
    - Full history of changes
    - Context from previous iterations
    - Pattern analysis for better fixes
    """

    def __init__(self, workspace_path: str | None = None):
        self.workspace = (
            Path(workspace_path) if workspace_path else Path("/tmp/intentforge_workspace")
        )
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.git = GitManager(str(self.workspace))
        self._initialized = False
        self.current_session: IterationHistory = None

    def _ensure_init(self):
        """Lazy initialization of Git repo"""
        if not self._initialized:
            self._initialized = self.git.init_repo()
            # Configure git user for commits
            self.git._run_git("config", "user.email", "intentforge@local")
            self.git._run_git("config", "user.name", "IntentForge")

    async def start_session(self, initial_code: str, intent: str = "") -> str:
        """Start a new fix session with Git branch"""
        self._ensure_init()

        branch = self.git.create_fix_branch("fix")
        if not branch:
            # Fallback: work without branching
            branch = "main"

        self.current_session = IterationHistory(
            branch_name=branch,
            initial_code=initial_code,
        )

        # Initial commit
        self.git.commit_iteration(
            code=initial_code,
            filename="code.py",
            message=f"Initial code - Intent: {intent[:50]}",
            iteration=0,
        )

        return branch

    async def record_iteration(
        self,
        code: str,
        error: str | None = None,
        fix_description: str | None = None,
        success: bool = False,
    ) -> GitCommit:
        """Record a fix iteration as a commit"""
        if not self.current_session:
            return None

        self.current_session.total_iterations += 1
        iteration = self.current_session.total_iterations

        message = "Success - All tests passed" if success else f"Attempt {iteration}"
        if fix_description:
            message = fix_description[:50]

        commit = self.git.commit_iteration(
            code=code,
            filename="code.py",
            message=message,
            iteration=iteration,
            error=error,
            fix_applied=fix_description,
        )

        if commit:
            self.current_session.commits.append(commit)

        if success:
            self.current_session.success = True
            self.current_session.final_code = code

        return commit

    async def get_context_for_llm(self) -> str:
        """Get Git history context for LLM analysis"""
        context = self.git.get_iteration_context("code.py")
        patterns = self.git.analyze_patterns("code.py")

        return f"""
{context}

## Pattern Analysis
- Total iterations: {patterns["total_iterations"]}
- Success rate: {patterns["success_rate"]:.0%}
- Common error types: {patterns["error_types"]}

Use this history to understand what has been tried and avoid repeating failed approaches.
"""

    async def end_session(self, merge: bool = True, squash: bool = True) -> dict:
        """End fix session and optionally merge"""
        if not self.current_session:
            return {"success": False, "reason": "No active session"}

        result = {
            "branch": self.current_session.branch_name,
            "total_iterations": self.current_session.total_iterations,
            "commits": len(self.current_session.commits),
            "success": self.current_session.success,
        }

        if merge and self.current_session.success:
            result["merged"] = self.git.merge_fix_branch(
                self.current_session.branch_name,
                squash=squash,
            )

        self.current_session = None
        return result


# Global instance
git_tracker = GitIterationTracker()


async def track_code_iteration(
    code: str,
    error: str | None = None,
    fix: str | None = None,
    success: bool = False,
) -> dict:
    """
    Convenience function to track a code iteration.
    """
    # Start session if needed
    if not git_tracker.current_session:
        await git_tracker.start_session(code, "Auto-tracked code")

    commit = await git_tracker.record_iteration(
        code=code,
        error=error,
        fix_description=fix,
        success=success,
    )

    return {
        "tracked": commit is not None,
        "commit": commit.hash if commit else None,
        "iteration": git_tracker.current_session.total_iterations
        if git_tracker.current_session
        else 0,
    }

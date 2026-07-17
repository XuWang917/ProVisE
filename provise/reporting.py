"""Terminal formatting and progress event reporting."""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, TextIO

CONCISE_OUTPUT_ENV = "PROVISE_CONCISE_OUTPUT"
_ANSI_CODES = {
    "bold": "\033[1m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "reset": "\033[0m",
}
_ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
_CONCISE_EVENTS = {
    "automatic_vlm_fallback_activated",
    "automatic_vlm_fallback_images_reused",
    "automatic_vlm_fallback_rejected",
    "automatic_vlm_fallback_smoke_started",
    "formal_evaluation_blocked",
    "evaluation_task_completed",
    "image_generation_result",
    "image_generation_retry_started",
    "ingestion_repair_completed",
    "ingestion_repair_failed",
    "ingestion_repair_started",
    "protocol_revision_completed",
    "protocol_revision_failed",
    "protocol_revision_skipped_external_failure",
    "protocol_revision_started",
    "run_stopped_existing_artifact",
    "smoke_generation_retry_started",
    "smoke_task_completed",
    "smoke_task_failed",
    "task_agent_decision",
    "task_agent_failed",
    "task_workflow_completed",
    "task_workflow_started",
}
_CONCISE_WAITING_PREFIXES = (
    "benchmark_ingestion_",
    "image_generation_",
    "protocol_agent_call_",
    "protocol_revision_agent_call_",
)
_CONCISE_HIDDEN_EVENTS = {"image_parse_result"}


def concise_output_enabled() -> bool:
    value = str(os.getenv(CONCISE_OUTPUT_ENV, "")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def detail_print(*values: Any, **kwargs: Any) -> None:
    if not concise_output_enabled():
        print(*values, **kwargs)


def terminal_colors_enabled(
    stream: TextIO | None = None,
    *,
    requested: bool | None = None,
) -> bool:
    if "NO_COLOR" in os.environ or os.getenv("CLICOLOR") == "0":
        return False
    if requested is False:
        return False
    if requested is True:
        return True
    if str(os.getenv("TERM", "")).strip().lower() == "dumb":
        return False
    if os.getenv("FORCE_COLOR") and os.getenv("FORCE_COLOR") != "0":
        return True
    target = stream or sys.stdout
    try:
        return bool(target.isatty())
    except (AttributeError, OSError):
        return False


def style_terminal(
    value: str,
    *,
    tone: str = "",
    bold: bool = False,
    stream: TextIO | None = None,
    enabled: bool | None = None,
) -> str:
    text = str(value)
    if not terminal_colors_enabled(stream, requested=enabled):
        return text
    codes = []
    if bold:
        codes.append(_ANSI_CODES["bold"])
    color = {
        "success": "green",
        "warning": "yellow",
        "error": "red",
    }.get(tone)
    if color:
        codes.append(_ANSI_CODES[color])
    if not codes:
        return text
    return "".join(codes) + text + _ANSI_CODES["reset"]


def terminal_width(value: str) -> int:
    return len(_ANSI_PATTERN.sub("", str(value)))


def display_task_name(task: str) -> str:
    value = str(task or "").strip()
    if "__" not in value:
        return value.replace("_", " ").title() if "_" in value else value
    parts = value.split("__")
    base = parts[0].replace("_", " ").title()
    schema = parts[1] if len(parts) > 1 else ""
    schema_label = {
        "binary_boolean": "boolean",
        "choice_selection": "multiple choice",
    }.get(schema, schema.replace("_", " "))
    return f"{base} ({schema_label})" if schema_label else base


def compact_path(path: str | Path, project_root: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    project = Path(project_root).expanduser().resolve()
    output_root = (project / "outputs").resolve()
    for root, prefix in ((output_root, Path("outputs")), (project, Path("."))):
        try:
            return str(prefix / resolved.relative_to(root))
        except ValueError:
            continue
    return str(resolved)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m" if seconds == 0 else f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


@dataclass
class ProgressReporter:
    """Terminal and JSONL progress reporting for long agent runs."""

    event_path: str | Path | None = None
    stream: TextIO = sys.stdout
    enabled: bool = True
    heartbeat_seconds: float = 1.0
    concise: bool | None = None
    color: bool | None = None
    _started_at: float = field(default_factory=time.monotonic, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _terminal_line_active: bool = field(default=False, init=False)
    _terminal_line_width: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.concise is None:
            self.concise = concise_output_enabled()
        if self.event_path:
            path = Path(self.event_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.event_path = path
        try:
            self.stream.reconfigure(line_buffering=True)
        except (AttributeError, OSError):
            pass

    def emit(
        self,
        message: str,
        *,
        event: str = "progress",
        status: str = "running",
        stage: int | None = None,
        total_stages: int | None = None,
        task: str = "",
        sample_id: str = "",
        terminal_transient: bool = False,
        **details: Any,
    ) -> None:
        payload: Dict[str, Any] = {
            "timestamp": _utc_now(),
            "elapsed_seconds": round(time.monotonic() - self._started_at, 3),
            "event": event,
            "status": status,
            "message": str(message),
        }
        if stage is not None:
            payload["stage"] = int(stage)
        if total_stages is not None:
            payload["total_stages"] = int(total_stages)
        if task:
            payload["task"] = str(task)
        if sample_id:
            payload["sample_id"] = str(sample_id)
        payload.update({key: value for key, value in details.items() if value is not None})

        prefix = ""
        if stage is not None and total_stages is not None:
            prefix = f"[{stage}/{total_stages}] "
        elif task:
            task_index = details.get("task_index")
            task_count = details.get("task_count")
            if task_index is not None and task_count is not None:
                prefix = (
                    f"[Task {int(task_index)}/{int(task_count)}: "
                    f"{display_task_name(task)}] "
                )
            else:
                prefix = f"[Task: {display_task_name(task)}] "
        tone = self._terminal_tone(event=event, status=status, details=details)
        line = style_terminal(
            prefix,
            bold=bool(prefix),
            stream=self.stream,
            enabled=self.color,
        ) + style_terminal(
            str(message),
            tone=tone,
            stream=self.stream,
            enabled=self.color,
        )

        with self._lock:
            if self.enabled and self._show_in_terminal(
                event=event,
                status=status,
                stage=stage,
            ):
                self._write_terminal(
                    line,
                    transient=terminal_transient or event.endswith("_heartbeat"),
                )
            if self.event_path:
                with Path(self.event_path).open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _show_in_terminal(
        self,
        *,
        event: str,
        status: str,
        stage: int | None,
    ) -> bool:
        if not self.concise:
            return True
        if event in _CONCISE_HIDDEN_EVENTS:
            return False
        if stage is not None or status in {"failed", "stopped"}:
            return True
        if event in _CONCISE_EVENTS:
            return True
        if event.startswith(_CONCISE_WAITING_PREFIXES):
            return True
        return False

    def _write_terminal(self, line: str, *, transient: bool) -> None:
        try:
            interactive = bool(self.stream.isatty())
        except (AttributeError, OSError):
            interactive = False

        if transient and interactive:
            visible_width = terminal_width(line)
            width = max(self._terminal_line_width, visible_width)
            self.stream.write("\r" + line + (" " * max(0, width - visible_width)))
            self.stream.flush()
            self._terminal_line_active = True
            self._terminal_line_width = width
            return

        if self._terminal_line_active:
            self.stream.write("\r" + (" " * self._terminal_line_width) + "\r")
            self._terminal_line_active = False
            self._terminal_line_width = 0
        print(line, file=self.stream, flush=True)

    @staticmethod
    def _terminal_tone(*, event: str, status: str, details: Dict[str, Any]) -> str:
        outcome = str(details.get("outcome") or "").lower()
        if event == "task_workflow_completed":
            outcome_tone = {
                "ready": "success",
                "deferred": "warning",
                "unresolved": "error",
            }.get(outcome)
            if outcome_tone:
                return outcome_tone
        if status == "failed":
            return "error"
        decision = str(details.get("decision") or "").lower()
        if decision in {"unsupported", "invalid"}:
            return "error"
        if status == "stopped" or decision == "fallback":
            return "warning"
        if any(token in event for token in ("fallback", "retry", "revision")):
            return "warning"
        if status in {"completed", "passed"}:
            return "success"
        return ""

    @contextmanager
    def waiting(
        self,
        message: str,
        *,
        event: str,
        task: str = "",
        heartbeat_seconds: float | None = None,
        **details: Any,
    ) -> Iterator[None]:
        interval = float(heartbeat_seconds or self.heartbeat_seconds)
        started = time.monotonic()
        stopped = threading.Event()
        self.emit(
            f"{message} (0s)",
            event=f"{event}_started",
            task=task,
            terminal_transient=True,
            elapsed_operation_seconds=0,
            **details,
        )

        def heartbeat() -> None:
            while not stopped.wait(max(0.5, interval)):
                elapsed = int(time.monotonic() - started)
                self.emit(
                    f"{message} ({format_elapsed(elapsed)})",
                    event=f"{event}_heartbeat",
                    task=task,
                    terminal_transient=True,
                    elapsed_operation_seconds=elapsed,
                    **details,
                )

        thread = threading.Thread(target=heartbeat, name=f"progress-{event}", daemon=True)
        thread.start()
        try:
            yield
        except Exception as exc:
            stopped.set()
            thread.join(timeout=1.0)
            self.emit(
                f"{message} failed: {type(exc).__name__}: {exc}",
                event=f"{event}_failed",
                status="failed",
                task=task,
                elapsed_operation_seconds=round(time.monotonic() - started, 3),
                **details,
            )
            raise
        else:
            stopped.set()
            thread.join(timeout=1.0)
            self.emit(
                f"{message} done ({format_elapsed(time.monotonic() - started)})",
                event=f"{event}_completed",
                status="completed",
                task=task,
                elapsed_operation_seconds=round(time.monotonic() - started, 3),
                **details,
            )


class NullProgressReporter(ProgressReporter):
    def __init__(self) -> None:
        super().__init__(event_path=None, enabled=False)

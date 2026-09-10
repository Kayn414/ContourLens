"""Machine-readable progress for the C++ GUI's background jobs
(gui/src/inference.{h,cpp}).

The GUI launches `uv run python -m source.<module> ...` with stdout+stderr on
a pipe and scans every line it reads:
- `@@progress {json}` drives its progress bar: {"stage", "current", "total",
  "fraction", "message"}; fraction is null for an indeterminate stage (e.g.
  waiting on a remote Modal call).
- `@@error {json}` carries a readable failure reason ({"message"}) to show
  in the GUI instead of just a non-zero exit code.
- `@@notice {json}` is something the user should know about that isn't a
  failure ({"message"}, e.g. "deployed the Modal app for you"); the GUI keeps
  it visible in the job's status after the job ends.
- Any other line is plain log output (shown in the job's log panel and
  echoed to the GUI's console).

Also parses tqdm's carriage-return progress output from a child process
(nnUNetv2_predict) into those progress events -- see TqdmProgressParser.
"""

from __future__ import annotations

import codecs
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass

PROGRESS_PREFIX = "@@progress "
ERROR_PREFIX = "@@error "
NOTICE_PREFIX = "@@notice "


def configure_stdio() -> None:
    """Call at the top of a GUI-launched entry point: stdout is a pipe there,
    so Python would default to the ANSI code page (and full buffering) --
    nnU-Net/tqdm output containing non-ASCII would then raise, and progress
    lines would arrive in bursts instead of live.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def emit(
    stage: str,
    current: int | None = None,
    total: int | None = None,
    message: str = "",
    fraction: float | None = None,
) -> None:
    if fraction is None and current is not None and total:
        fraction = current / total
    if fraction is not None:
        fraction = min(max(fraction, 0.0), 1.0)
    payload = {"stage": stage, "current": current, "total": total, "fraction": fraction, "message": message}
    print(PROGRESS_PREFIX + json.dumps(payload), flush=True)


def emit_error(message: str) -> None:
    print(ERROR_PREFIX + json.dumps({"message": message}), flush=True)


def emit_notice(message: str) -> None:
    print(NOTICE_PREFIX + json.dumps({"message": message}), flush=True)


@dataclass
class ProgressEvent:
    stage: str
    current: int | None
    total: int | None
    fraction: float | None
    message: str


# tqdm's default bar ends with "<n>/<total> [elapsed<remaining, rate]".
_TQDM_RE = re.compile(r"(\d+)/(\d+)\s*\[")


class TqdmProgressParser:
    """Turns a child process's raw output into progress events + plain lines.

    nnUNetv2_predict draws one sliding-window tqdm bar per fold (see
    nnunetv2/inference/predict_from_raw_data.py: predict_logits_from_preprocessed_data
    loops over list_of_parameters, each pass a fresh tqdm(total=len(slicers))),
    so a bar whose count drops back down marks the start of the next pass,
    and overall fraction = (pass + current/total) / n_passes. Its final
    "Collecting results" bar is segmentation export, reported as its own stage.
    """

    def __init__(self, n_passes: int, stage: str = "predicting", pass_label: str = "Fold"):
        self.n_passes = max(1, n_passes)
        self.stage = stage
        self.pass_label = pass_label
        self.pass_index = -1
        self._last: tuple[int, int] | None = None
        self._buffer = ""

    def feed(self, text: str) -> tuple[list[ProgressEvent], list[str]]:
        self._buffer += text
        *segments, self._buffer = re.split(r"[\r\n]", self._buffer)
        return self._parse_segments(segments)

    def flush(self) -> tuple[list[ProgressEvent], list[str]]:
        segments, self._buffer = [self._buffer], ""
        return self._parse_segments(segments)

    def _parse_segments(self, segments: list[str]) -> tuple[list[ProgressEvent], list[str]]:
        events: list[ProgressEvent] = []
        lines: list[str] = []
        for segment in segments:
            if not segment.strip():
                continue
            match = _TQDM_RE.search(segment)
            if match is None:
                lines.append(segment)
                continue

            current, total = int(match.group(1)), int(match.group(2))
            if "Collecting results" in segment:
                events.append(ProgressEvent("exporting", current, total, current / total if total else None,
                                            f"Collecting results {current}/{total}"))
                continue

            if self._last is None or current < self._last[0] or total != self._last[1]:
                self.pass_index += 1
            self._last = (current, total)

            shown_pass = min(self.pass_index, self.n_passes - 1)
            within = current / total if total else 0.0
            fraction = min((shown_pass + within) / self.n_passes, 1.0)
            message = f"{self.pass_label} {shown_pass + 1}/{self.n_passes} | patch {current}/{total}"
            events.append(ProgressEvent(self.stage, current, total, fraction, message))
        return events, lines


def run_with_progress(
    cmd: list[str], parser: TqdmProgressParser, env: dict[str, str] | None = None, report_progress: bool = True
) -> None:
    """subprocess.run(cmd, check=True) equivalent that turns the child's tqdm
    bars into @@progress lines (unless `report_progress` is off) and forwards
    its other output line by line.
    """
    child_env = {**(env if env is not None else os.environ), "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=child_env)
    assert proc.stdout is not None
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def forward(events: list[ProgressEvent], lines: list[str]) -> None:
        for line in lines:
            print(line, flush=True)
        if events and report_progress:  # only the latest state matters for a progress bar
            e = events[-1]
            emit(e.stage, e.current, e.total, e.message, fraction=e.fraction)

    while chunk := os.read(proc.stdout.fileno(), 4096):  # returns as soon as any bytes arrive; b"" at EOF
        forward(*parser.feed(decoder.decode(chunk)))
    forward(*parser.feed(decoder.decode(b"", final=True)))
    forward(*parser.flush())

    returncode = proc.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


def run_forwarding_output(cmd: list[str], env: dict[str, str] | None = None) -> None:
    """subprocess.run(cmd, check=True) whose output reaches the GUI's job log
    live, line by line, for commands without a meaningful tqdm bar (e.g.
    `modal deploy`) -- the caller reports its own stage instead."""
    run_with_progress(cmd, TqdmProgressParser(n_passes=1), env=env, report_progress=False)

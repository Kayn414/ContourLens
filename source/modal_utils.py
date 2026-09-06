"""Shared Modal infrastructure for the nnU-Net training pipeline (used by
source/modal_nnunet.py).

Example usage:

from source.modal_utils import app, quote_command, submit_commands

def build_run_commands(args):
    return [
        ["nnUNetv2_train", "501", "3d_fullres", str(fold), "--npz"]
        for fold in args.folds
    ]

@app.local_entrypoint()
def modal_main(*argv: str) -> None:
    args = make_parser().parse_args(list(argv))
    commands = build_run_commands(args)
    submit_commands(commands)
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import PurePosixPath

import modal

GPU = "A10G"
MAX_CONTAINERS = 5
RUN_TIMEOUT_SECONDS = 24 * 60 * 60


app = modal.App("nnunet-training")

RAW_VOLUME_NAME = "nnunet-raw-volume"
PREPROCESSED_VOLUME_NAME = "nnunet-preprocessed"
RESULTS_VOLUME_NAME = "nnunet-results-volume"

raw_volume = modal.Volume.from_name(RAW_VOLUME_NAME, create_if_missing=True)
preprocessed_vol = modal.Volume.from_name(PREPROCESSED_VOLUME_NAME, create_if_missing=True)
results_vol = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)

VOL_RAW = PurePosixPath("/raw")
VOL_PREPROCESSED = PurePosixPath("/processed")
VOL_RESULTS = PurePosixPath("/results")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .uv_sync()
    .env({
        "nnUNet_raw": str(VOL_RAW),
        "nnUNet_preprocessed": str(VOL_PREPROCESSED),
        "nnUNet_results": str(VOL_RESULTS),
        "nnUNet_n_proc_DA": "8",
    })
    .add_local_python_source("source")
)


def quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


@app.function(
    image=image,
    gpu=GPU,
    volumes={VOL_RAW: raw_volume, VOL_PREPROCESSED: preprocessed_vol, VOL_RESULTS: results_vol},
    timeout=RUN_TIMEOUT_SECONDS,
    max_containers=MAX_CONTAINERS,
    # Auto-restart a container that dies to a preemption / infra blip. Safe for
    # `nnUNetv2_train ... --c` (each retry just resumes from checkpoint_latest);
    # a retry of a *fresh* train command restarts that fold from scratch, so
    # prefer launching long runs with --continue-training once a checkpoint exists.
    retries=modal.Retries(max_retries=2, backoff_coefficient=1.0, initial_delay=10.0),
)
def run_command(command: list[str]) -> str:
    command_str = quote_command(command)
    print(command_str, flush=True)
    subprocess.run(command, check=True)
    preprocessed_vol.commit()
    results_vol.commit()
    return command_str


def submit_commands(commands: list[list[str]]) -> None:
    """Spawn each command as an independent, server-owned Modal job.

    Uses `.spawn()` rather than `.map()` on purpose: `.map()` inputs are owned
    by the local client's function-call context, so if the `modal run` process
    dies (disconnect, OOM-kill, reboot) Modal cancels the in-flight inputs --
    even under `--detach`, which only keeps the *app* alive, not a client-owned
    map call. `.spawn()` returns detached FunctionCalls that keep running to
    completion regardless of the client. Still launch with `--detach` so the
    ephemeral app itself isn't torn down when the client disconnects.
    """
    print(
        f"Spawning {len(commands)} Modal jobs "
        f"with max_containers={MAX_CONTAINERS}, gpu={GPU}, "
        f"timeout={RUN_TIMEOUT_SECONDS}s.",
        flush=True,
    )
    calls = []
    for command in commands:
        fc = run_command.spawn(command)
        calls.append((command, fc))
        print(f"  spawned {fc.object_id}: {quote_command(command)}", flush=True)

    failures = []
    for command, fc in calls:
        command_str = quote_command(command)
        try:
            result = fc.get()
            print(f"Completed: {result}", flush=True)
        except BaseException as exc:  # noqa: BLE001 - report and continue to next job
            print(f"Failed: {command_str}", flush=True)
            print(f"Error: {exc!r}", flush=True)
            failures.append(command_str)

    if failures:
        print(f"{len(failures)} of {len(commands)} Modal jobs failed.", flush=True)
        raise SystemExit(1)

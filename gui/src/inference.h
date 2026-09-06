#pragma once

// Runs source/run_inference_for_gui.py (nnU-Net predict + export) as a
// detached child process, polled once per frame so the UI thread never
// blocks -- inference is a multi-minute GPU job. See gui/CMakeLists.txt for
// DICOM_RT_REPO_DIR (the child's working directory, so `uv run python -m
// source...` and its relative data/ paths resolve regardless of where the
// GUI exe itself lives).

#include <windows.h>
#include <string>

struct InferenceJob
{
    bool running = false;
    bool has_result = false; // true once PollInferenceJob reports completion, until the caller consumes it
    HANDLE process_handle = nullptr;
    DWORD exit_code = 0;
    DWORD start_tick_ms = 0;
};

// Launches the child. Returns false (logging to stderr) if CreateProcess
// fails outright; does nothing if a job is already running.
bool StartInferenceJob(InferenceJob& job, const std::string& repo_dir);

// Non-blocking: returns true exactly once, on the frame the job finishes
// (check job.exit_code then). Returns false while running or idle.
bool PollInferenceJob(InferenceJob& job);

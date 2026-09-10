#pragma once

// Runs a Python module (`uv run python -m <module> <args...>`) from the repo
// root as a child process, polled once per frame so the UI thread never
// blocks -- inference is a multi-minute GPU job, and even loading a dropped
// CT (source/gui_load.py) takes seconds. The child's stdout+stderr come back
// over a pipe: "@@progress {json}" / "@@error {json}" lines (see
// source/gui_progress.py) drive the progress bar and error text, and every
// other line is kept as a log tail and echoed to the GUI's own console. See
// gui/CMakeLists.txt for CONTOURLENS_REPO_DIR (the child's working directory, so
// `uv run python -m source...` and its relative data/ paths resolve
// regardless of where the GUI exe itself lives).

#include <windows.h>
#include <deque>
#include <string>
#include <vector>

enum class JobKind { None, LoadCt, LoadMasks, LoadDose, Inference, Hd95, Export, Batch };

struct JobProgress
{
    std::string stage;      // "reading", "staging", "predicting", "exporting", ...
    std::string message;    // e.g. "Fold 2/4 | patch 118/240"
    float fraction = -1.0f; // [0, 1]; negative while indeterminate
};

struct BackgroundJob
{
    // Set by the caller before StartJob; read back on completion to decide what to reload.
    JobKind kind = JobKind::None;
    std::string title;     // "Inference", "Loading CT", ...
    std::string case_name; // the case (data/<case>/gui_export) the job writes into
    std::string slot;      // LoadMasks: "gt" | "prediction"
    std::string source;    // LoadCt: the dropped path (becomes the case's source_ct)
    std::string output_dir; // Export/Batch: the folder the job writes into (for "Open folder")

    bool running = false;
    bool has_result = false; // true once PollJob reports completion
    bool cancelled = false;
    DWORD exit_code = 0;
    DWORD start_tick_ms = 0;
    DWORD end_tick_ms = 0;

    JobProgress progress;
    std::string error_message;   // from the child's last "@@error" line, if any
    std::vector<std::string> notices; // from "@@notice" lines (e.g. "deployed the Modal app"); kept visible after the job ends
    std::deque<std::string> log; // last kMaxLogLines plain output lines
    bool log_updated = false;    // set whenever `log` grows; the log view clears it after auto-scrolling
    static constexpr size_t kMaxLogLines = 300;

    HANDLE process_handle = nullptr;
    HANDLE job_object = nullptr; // holds the whole uv -> python -> nnUNetv2_predict tree, for CancelJob
    HANDLE stdout_read = nullptr;
    std::string pending; // bytes read past the last newline
};

// Launches the child; `args` are UTF-8 and quoted as needed. Resets the
// job's progress/log/error state. Returns false (with job.error_message set)
// if the process can't be started; does nothing if a job is already running.
bool StartJob(BackgroundJob& job, const std::string& repo_dir, const std::string& module, const std::vector<std::string>& args);

// Non-blocking: drains any child output, and returns true exactly once, on
// the frame the job finishes (check job.exit_code / job.cancelled then).
// Returns false while running or idle. Call every frame, even when the
// window is minimized, or the child can block on a full pipe.
bool PollJob(BackgroundJob& job);

// Kills the child process tree; PollJob reports completion on a later frame.
void CancelJob(BackgroundJob& job);

// Closes the job's handles without killing it (app exit: a still-running job finishes on its own).
void ReleaseJob(BackgroundJob& job);

std::wstring Utf8ToWide(const std::string& s);
std::string WideToUtf8(const std::wstring& w);

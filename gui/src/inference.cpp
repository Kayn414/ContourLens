#include "inference.h"

#include <cstdio>
#include <vector>

bool StartInferenceJob(InferenceJob& job, const std::string& repo_dir, const std::string& case_name)
{
    if (job.running)
        return false;

    // Widen repo_dir/case_name (ASCII in practice) without pulling in <filesystem>.
    std::wstring cwd(repo_dir.begin(), repo_dir.end());
    std::wstring case_name_w(case_name.begin(), case_name.end());

    // Runs in the same console as the GUI (no CREATE_NEW_CONSOLE) so nnU-Net's
    // (verbose) progress output is visible somewhere while the job runs.
    std::wstring command = L"cmd /c \"uv run python -m source.run_inference_for_gui " + case_name_w + L"\"";
    std::vector<wchar_t> command_buf(command.begin(), command.end());
    command_buf.push_back(L'\0');

    STARTUPINFOW startup_info = {};
    startup_info.cb = sizeof(startup_info);
    PROCESS_INFORMATION process_info = {};

    BOOL ok = CreateProcessW(
        nullptr, command_buf.data(), nullptr, nullptr, FALSE, 0, nullptr,
        cwd.c_str(), &startup_info, &process_info);
    if (!ok)
    {
        fprintf(stderr, "StartInferenceJob: CreateProcess failed (error %lu)\n", GetLastError());
        return false;
    }

    CloseHandle(process_info.hThread);
    job.process_handle = process_info.hProcess;
    job.running = true;
    job.has_result = false;
    job.start_tick_ms = GetTickCount();
    return true;
}

bool PollInferenceJob(InferenceJob& job)
{
    if (!job.running)
        return false;

    if (WaitForSingleObject(job.process_handle, 0) != WAIT_OBJECT_0)
        return false;

    GetExitCodeProcess(job.process_handle, &job.exit_code);
    CloseHandle(job.process_handle);
    job.process_handle = nullptr;
    job.running = false;
    job.has_result = true;
    return true;
}

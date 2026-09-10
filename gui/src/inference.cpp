#include "inference.h"

#include <algorithm>
#include <cstdio>

#include <nlohmann/json.hpp>

std::wstring Utf8ToWide(const std::string& s)
{
    if (s.empty())
        return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), n);
    return w;
}

std::string WideToUtf8(const std::wstring& w)
{
    if (w.empty())
        return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
}

// One argv entry, quoted per the MSVC CRT's command-line parsing rules
// (which Python uses): backslashes are literal except before a quote.
static std::wstring QuoteArg(const std::wstring& arg)
{
    if (!arg.empty() && arg.find_first_of(L" \t\"") == std::wstring::npos)
        return arg;

    std::wstring out = L"\"";
    size_t backslashes = 0;
    for (wchar_t c : arg)
    {
        if (c == L'\\')
        {
            ++backslashes;
            continue;
        }
        out.append(c == L'"' ? backslashes * 2 + 1 : backslashes, L'\\');
        out += c;
        backslashes = 0;
    }
    out.append(backslashes * 2, L'\\');
    out += L'"';
    return out;
}

static void AppendLog(BackgroundJob& job, const std::string& line)
{
    printf("%s\n", line.c_str()); // keep the GUI's console as useful as when the child wrote to it directly
    fflush(stdout);               // live even when the GUI's stdout is redirected to a file (fully buffered otherwise)
    job.log.push_back(line);
    while (job.log.size() > BackgroundJob::kMaxLogLines)
        job.log.pop_front();
    job.log_updated = true;
}

static void HandleLine(BackgroundJob& job, std::string line)
{
    // Carriage-return redraws (progress spinners) within one line: keep the latest state.
    while (!line.empty() && line.back() == '\r')
        line.pop_back();
    size_t cr = line.rfind('\r');
    if (cr != std::string::npos)
        line.erase(0, cr + 1);
    if (line.empty())
        return;

    static const std::string kProgress = "@@progress ";
    static const std::string kError = "@@error ";
    static const std::string kNotice = "@@notice ";
    try
    {
        if (line.compare(0, kProgress.size(), kProgress) == 0)
        {
            nlohmann::json j = nlohmann::json::parse(line.substr(kProgress.size()));
            job.progress.stage = j.value("stage", std::string());
            job.progress.message = j.value("message", std::string());
            job.progress.fraction = (j.contains("fraction") && j["fraction"].is_number()) ? j["fraction"].get<float>() : -1.0f;
            return;
        }
        if (line.compare(0, kError.size(), kError) == 0)
        {
            nlohmann::json j = nlohmann::json::parse(line.substr(kError.size()));
            job.error_message = j.value("message", std::string());
            AppendLog(job, "ERROR: " + job.error_message);
            return;
        }
        if (line.compare(0, kNotice.size(), kNotice) == 0)
        {
            nlohmann::json j = nlohmann::json::parse(line.substr(kNotice.size()));
            job.notices.push_back(j.value("message", std::string()));
            AppendLog(job, "NOTE: " + job.notices.back());
            return;
        }
    }
    catch (const nlohmann::json::exception&)
    {
        // Malformed protocol line: fall through and show it verbatim.
    }
    AppendLog(job, line);
}

static void DrainPipe(BackgroundJob& job)
{
    if (!job.stdout_read)
        return;

    char buf[4096];
    for (int chunk = 0; chunk < 64; ++chunk) // bounded, so a flood of output can't stall a frame
    {
        DWORD available = 0;
        if (!PeekNamedPipe(job.stdout_read, nullptr, 0, nullptr, &available, nullptr) || available == 0)
            break;
        DWORD bytes_read = 0;
        if (!ReadFile(job.stdout_read, buf, std::min<DWORD>(available, sizeof(buf)), &bytes_read, nullptr) || bytes_read == 0)
            break;

        job.pending.append(buf, bytes_read);
        size_t start = 0, newline;
        while ((newline = job.pending.find('\n', start)) != std::string::npos)
        {
            HandleLine(job, job.pending.substr(start, newline - start));
            start = newline + 1;
        }
        job.pending.erase(0, start);
    }
}

static void CloseJobHandles(BackgroundJob& job)
{
    if (job.stdout_read) { CloseHandle(job.stdout_read); job.stdout_read = nullptr; }
    if (job.process_handle) { CloseHandle(job.process_handle); job.process_handle = nullptr; }
    if (job.job_object) { CloseHandle(job.job_object); job.job_object = nullptr; }
}

bool StartJob(BackgroundJob& job, const std::string& repo_dir, const std::string& module, const std::vector<std::string>& args)
{
    if (job.running)
        return false;

    job.has_result = false;
    job.cancelled = false;
    job.exit_code = 0;
    job.progress = JobProgress{};
    job.progress.message = "Starting...";
    job.error_message.clear();
    job.notices.clear();
    job.log.clear();
    job.pending.clear();

    static bool process_env_configured = false;
    if (!process_env_configured)
    {
        // Inherited by every child: unbuffered, UTF-8 Python output over the pipe.
        SetEnvironmentVariableW(L"PYTHONUNBUFFERED", L"1");
        SetEnvironmentVariableW(L"PYTHONIOENCODING", L"utf-8");
        SetConsoleOutputCP(CP_UTF8); // echoed child output is UTF-8
        process_env_configured = true;
    }

    SECURITY_ATTRIBUTES inheritable = { sizeof(inheritable), nullptr, TRUE };
    HANDLE read_end = nullptr, write_end = nullptr;
    if (!CreatePipe(&read_end, &write_end, &inheritable, 1 << 16))
    {
        job.error_message = "CreatePipe failed (error " + std::to_string(GetLastError()) + ")";
        return false;
    }
    SetHandleInformation(read_end, HANDLE_FLAG_INHERIT, 0); // only the child's end is inherited
    HANDLE nul_input = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &inheritable, OPEN_EXISTING, 0, nullptr);

    STARTUPINFOW startup_info = {};
    startup_info.cb = sizeof(startup_info);
    startup_info.dwFlags = STARTF_USESTDHANDLES;
    startup_info.hStdInput = nul_input;
    startup_info.hStdOutput = write_end;
    startup_info.hStdError = write_end;

    std::wstring command = L"uv run python -m " + Utf8ToWide(module);
    for (const std::string& arg : args)
        command += L" " + QuoteArg(Utf8ToWide(arg));
    std::wstring cwd = Utf8ToWide(repo_dir);

    // Suspended until it's in the job object, so nothing it spawns escapes
    // CancelJob. Shares the GUI's console (no CREATE_NO_WINDOW): console-less,
    // every process nnU-Net spawns would pop up a console window of its own.
    PROCESS_INFORMATION process_info = {};
    auto launch = [&](std::wstring command_line)
    {
        std::vector<wchar_t> buf(command_line.begin(), command_line.end());
        buf.push_back(L'\0');
        return CreateProcessW(nullptr, buf.data(), nullptr, nullptr, TRUE, CREATE_SUSPENDED,
            nullptr, cwd.c_str(), &startup_info, &process_info);
    };
    BOOL ok = launch(command);
    if (!ok && GetLastError() == ERROR_FILE_NOT_FOUND)
        ok = launch(L"cmd /s /c \"" + command + L"\""); // uv only reachable as a .cmd/.bat shim on PATH

    DWORD launch_error = GetLastError();
    CloseHandle(write_end); // the child holds its own copy; ReadFile sees EOF once it exits
    if (nul_input != INVALID_HANDLE_VALUE)
        CloseHandle(nul_input);
    if (!ok)
    {
        CloseHandle(read_end);
        job.error_message = "Couldn't start `uv run python -m " + module + "` (CreateProcess error " +
            std::to_string(launch_error) + ") -- is uv on PATH?";
        fprintf(stderr, "StartJob: %s\n", job.error_message.c_str());
        return false;
    }

    job.job_object = CreateJobObjectW(nullptr, nullptr);
    if (job.job_object && !AssignProcessToJobObject(job.job_object, process_info.hProcess))
    {
        CloseHandle(job.job_object);
        job.job_object = nullptr; // Cancel falls back to killing just the direct child
    }
    ResumeThread(process_info.hThread);
    CloseHandle(process_info.hThread);

    job.process_handle = process_info.hProcess;
    job.stdout_read = read_end;
    job.running = true;
    job.start_tick_ms = GetTickCount();
    printf("Started job: %s\n", WideToUtf8(command).c_str());
    return true;
}

bool PollJob(BackgroundJob& job)
{
    if (!job.running)
        return false;

    DrainPipe(job);
    if (WaitForSingleObject(job.process_handle, 0) != WAIT_OBJECT_0)
        return false;

    DrainPipe(job); // output written between the last drain and exit
    if (!job.pending.empty())
    {
        HandleLine(job, job.pending);
        job.pending.clear();
    }

    GetExitCodeProcess(job.process_handle, &job.exit_code);
    CloseJobHandles(job);
    job.running = false;
    job.has_result = true;
    job.end_tick_ms = GetTickCount();
    return true;
}

void CancelJob(BackgroundJob& job)
{
    if (!job.running)
        return;
    job.cancelled = true;
    if (job.job_object)
        TerminateJobObject(job.job_object, 1);
    else
        TerminateProcess(job.process_handle, 1);
}

void ReleaseJob(BackgroundJob& job)
{
    CloseJobHandles(job);
    job.running = false;
}

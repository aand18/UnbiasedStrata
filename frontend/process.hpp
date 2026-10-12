// frontend/process.hpp - the engine's child process and the two pipes to it, on POSIX and on Windows.
//
// POSIX: fork + execv of this program, as before.  Windows: CreateProcessW with an explicit list of the handles the
// child inherits (only its own pipe ends and stderr: a child holding the parent's end of its stdin would never see
// end of file), started suspended and put in a Job Object that kills it when the front end's handle closes - so an
// engine cannot outlive a front end that crashed, holding the GPU.
#pragma once
#include <cstddef>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstdio>
#else
#include "child_watch.hpp"
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace strata_frontend {

#if defined(_WIN32)
inline std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), nullptr, 0);
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), w.data(), n);
    return w;
}

inline std::string narrow(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int) w.size(), nullptr, 0, nullptr, nullptr);
    std::string s((size_t) n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int) w.size(), s.data(), n, nullptr, nullptr);
    return s;
}

// One argument, quoted so that the C runtime's command-line parser (CommandLineToArgvW's rules) gives it back as it
// was: backslashes are literal except before a quote.
inline void append_arg(std::wstring& cmd, const std::wstring& a) {
    if (!cmd.empty()) cmd += L' ';
    if (!a.empty() && a.find_first_of(L" \t\n\v\"") == std::wstring::npos) { cmd += a; return; }
    cmd += L'"';
    for (size_t i = 0;; ++i) {
        size_t slashes = 0;
        while (i < a.size() && a[i] == L'\\') { ++i; ++slashes; }
        if (i == a.size()) { cmd.append(slashes * 2, L'\\'); break; }
        if (a[i] == L'"') cmd.append(slashes * 2 + 1, L'\\');
        else cmd.append(slashes, L'\\');
        cmd += a[i];
    }
    cmd += L'"';
}

inline std::string exit_reason(DWORD code) {
    char buf[96];
    if (code >= 0xC0000000u) std::snprintf(buf, sizeof buf, "engine crashed (exception 0x%08lX)", (unsigned long) code);
    else std::snprintf(buf, sizeof buf, "engine exited with code %lu", (unsigned long) code);
    return buf;
}
#endif

class ChildProcess {
public:
    ChildProcess() = default;
    ChildProcess(const ChildProcess&) = delete;
    ChildProcess& operator=(const ChildProcess&) = delete;

#if defined(_WIN32)
    ~ChildProcess() {
        close_stdin();
        for (HANDLE* h : {&out_, &proc_}) if (*h != nullptr) { CloseHandle(*h); *h = nullptr; }
        // job_ stays open: closing it would kill the engine, and the front end may still be shutting it down
    }

    /// `args[0]` is the child's argv[0]; an empty `executable` is this program.
    bool start(const std::vector<std::string>& args, const std::string& executable, std::string& err) {
        SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
        HANDLE in_r = nullptr, in_w = nullptr, out_r = nullptr, out_w = nullptr;
        if (!CreatePipe(&in_r, &in_w, &sa, 0) || !CreatePipe(&out_r, &out_w, &sa, 0)) {
            err = "CreatePipe failed (error " + std::to_string(GetLastError()) + ")";
            return false;
        }
        SetHandleInformation(in_w, HANDLE_FLAG_INHERIT, 0);
        SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0);
        // the engine's messages go where the front end's do; an inheritable copy, as the handle list requires
        HANDLE err_h = GetStdHandle(STD_ERROR_HANDLE), err_dup = nullptr;
        if (err_h != nullptr && err_h != INVALID_HANDLE_VALUE)
            DuplicateHandle(GetCurrentProcess(), err_h, GetCurrentProcess(), &err_dup, 0, TRUE, DUPLICATE_SAME_ACCESS);
        HANDLE inherit[3] = {in_r, out_w, err_dup};
        const DWORD n_inherit = err_dup != nullptr ? 3 : 2;

        SIZE_T size = 0;
        InitializeProcThreadAttributeList(nullptr, 1, 0, &size);
        std::vector<char> attr_buf(size);
        auto* attrs = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attr_buf.data());
        STARTUPINFOEXW si{};
        si.StartupInfo.cb = sizeof si;
        si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
        si.StartupInfo.hStdInput = in_r;
        si.StartupInfo.hStdOutput = out_w;
        si.StartupInfo.hStdError = err_dup;
        bool ok = InitializeProcThreadAttributeList(attrs, 1, 0, &size) &&
                  UpdateProcThreadAttribute(attrs, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherit,
                                            n_inherit * sizeof(HANDLE), nullptr, nullptr);
        si.lpAttributeList = attrs;

        std::wstring app;
        if (executable.empty()) {
            app.resize(32768);
            app.resize(GetModuleFileNameW(nullptr, app.data(), (DWORD) app.size()));
        } else {
            app = widen(executable);
        }
        std::wstring cmd;
        for (const std::string& a : args) append_arg(cmd, widen(a));

        job_ = CreateJobObjectW(nullptr, nullptr);
        if (job_ != nullptr) {
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
            li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            SetInformationJobObject(job_, JobObjectExtendedLimitInformation, &li, sizeof li);
        }
        PROCESS_INFORMATION pi{};
        ok = ok && CreateProcessW(app.c_str(), cmd.data(), nullptr, nullptr, TRUE,
                                  CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT, nullptr, nullptr,
                                  &si.StartupInfo, &pi);
        const DWORD create_error = GetLastError();
        DeleteProcThreadAttributeList(attrs);
        for (HANDLE h : {in_r, out_w, err_dup}) if (h != nullptr) CloseHandle(h);
        if (!ok) {
            CloseHandle(in_w);
            CloseHandle(out_r);
            err = "could not start the engine (CreateProcessW error " + std::to_string(create_error) + ")";
            return false;
        }
        if (job_ == nullptr || !AssignProcessToJobObject(job_, pi.hProcess))
            std::fprintf(stderr, "unbiased-strata: WARNING: the engine is not tied to this process (error %lu): "
                                 "if the front end crashes, end strata's engine process by hand\n", GetLastError());
        ResumeThread(pi.hThread);
        CloseHandle(pi.hThread);
        proc_ = pi.hProcess;
        in_ = in_w;
        out_ = out_r;
        return true;
    }

    /// Bytes written; 0 or less once the child's stdin is closed.
    long write(const char* p, size_t n) {
        DWORD done = 0;
        if (in_ == nullptr || !WriteFile(in_, p, (DWORD) (n < (1u << 30) ? n : (1u << 30)), &done, nullptr)) return -1;
        return (long) done;
    }

    /// Bytes read; 0 at the end of the child's output, less than 0 on an error.
    long read(char* p, size_t n) {
        for (;;) {
            DWORD got = 0;
            if (!ReadFile(out_, p, (DWORD) (n < (1u << 30) ? n : (1u << 30)), &got, nullptr))
                return GetLastError() == ERROR_BROKEN_PIPE ? 0 : -1;
            if (got > 0) return (long) got;
        }
    }

    bool started() const { return proc_ != nullptr && !exited_; }
    unsigned long id() const { return proc_ != nullptr ? (unsigned long) GetProcessId(proc_) : 0; }

    /// False once the child has exited, with the reason in `why`.
    bool running(std::string& why) {
        if (proc_ == nullptr) { why = "engine not started"; return false; }
        if (WaitForSingleObject(proc_, 0) == WAIT_TIMEOUT) return true;
        DWORD code = 0;
        GetExitCodeProcess(proc_, &code);
        why = exit_reason(code);
        exited_ = true;
        return false;
    }

    void close_stdin() {
        if (in_ != nullptr) { CloseHandle(in_); in_ = nullptr; }
    }

    /// True when the child exited within `ms`.
    bool wait(int ms) {
        if (proc_ == nullptr || WaitForSingleObject(proc_, (DWORD) ms) == WAIT_OBJECT_0) { exited_ = true; return true; }
        return false;
    }

    void kill() {
        if (proc_ == nullptr) return;
        TerminateProcess(proc_, 1);
        WaitForSingleObject(proc_, 5000);
        exited_ = true;
    }

private:
    HANDLE proc_ = nullptr, job_ = nullptr, in_ = nullptr, out_ = nullptr;
    bool exited_ = false;
#else
    ~ChildProcess() {
        close_stdin();
        if (out_fd_ >= 0) ::close(out_fd_);
    }

    /// `args[0]` is the child's argv[0]; an empty `executable` is this program.
    bool start(const std::vector<std::string>& args, const std::string& executable, std::string& err) {
        int to_child[2], from_child[2];
        if (::pipe(to_child) != 0 || ::pipe(from_child) != 0) { err = "pipe failed"; return false; }
        const std::string exe = executable.empty() ? "/proc/self/exe" : executable;
        pid_ = ::fork();
        if (pid_ < 0) { err = "fork failed"; return false; }
        if (pid_ == 0) {
            ::dup2(to_child[0], STDIN_FILENO);
            ::dup2(from_child[1], STDOUT_FILENO);
            ::close(to_child[0]); ::close(to_child[1]); ::close(from_child[0]); ::close(from_child[1]);
            std::vector<char*> argv;
            for (const std::string& a : args) argv.push_back(const_cast<char*>(a.c_str()));
            argv.push_back(nullptr);
            ::execv(exe.c_str(), argv.data());
            std::fprintf(stderr, "unbiased-strata: could not start the engine: %s\n", std::strerror(errno));
            ::_exit(127);
        }
        ::close(to_child[0]);
        ::close(from_child[1]);
        in_fd_ = to_child[1];
        out_fd_ = from_child[0];
        return true;
    }

    /// Bytes written; 0 or less once the child's stdin is closed.
    long write(const char* p, size_t n) {
        for (;;) {
            const ssize_t w = in_fd_ >= 0 ? ::write(in_fd_, p, n) : -1;
            if (w < 0 && errno == EINTR) continue;
            return (long) w;
        }
    }

    /// Bytes read; 0 at the end of the child's output, less than 0 on an error.
    long read(char* p, size_t n) {
        for (;;) {
            const ssize_t r = ::read(out_fd_, p, n);
            if (r < 0 && errno == EINTR) continue;
            return (long) r;
        }
    }

    bool started() const { return pid_ > 0; }
    unsigned long id() const { return pid_ > 0 ? (unsigned long) pid_ : 0; }

    /// False once the child has exited (and is reaped), with the reason in `why`.
    bool running(std::string& why) {
        if (pid_ > 0 && child_running(pid_, why_)) return true;
        why = why_.empty() ? "engine not started" : why_;
        return false;
    }

    void close_stdin() {
        if (in_fd_ >= 0) { ::close(in_fd_); in_fd_ = -1; }
    }

    /// True when the child exited (and is reaped) within `ms`.
    bool wait(int ms) {
        for (int waited = 0;; waited += 100) {
            if (pid_ <= 0 || !child_running(pid_, why_)) return true;
            if (waited >= ms) return false;
            ::usleep(100 * 1000);
        }
    }

    void kill() {
        if (pid_ <= 0) return;
        ::kill(pid_, SIGKILL);
        while (pid_ > 0 && child_running(pid_, why_)) ::usleep(10 * 1000);
    }

private:
    pid_t pid_ = -1;
    int in_fd_ = -1, out_fd_ = -1;
    std::string why_;       // how the child ended, once it is reaped
#endif
};

}  // namespace strata_frontend

# Linux status

Native Linux inference is not supported by this snapshot yet. The Vulkan math
is portable, but the budgeted expert store intentionally uses Windows
`OVERLAPPED`/unbuffered I/O, `VirtualAlloc`, and Windows process-memory APIs.
Pretending that the current source is a supported Linux build would be unsafe.

## Contributor setup

On Ubuntu 24.04 or a current equivalent:

```bash
sudo apt update
sudo apt install clang cmake ninja-build glslang-tools libvulkan-dev vulkan-tools python3
vulkaninfo --summary
```

The native port needs three bounded replacements before `CMakeLists.txt` should
enable Linux:

1. `dlopen`/`dlsym` Vulkan loading in place of `LoadLibrary`/`GetProcAddress`;
2. aligned `O_DIRECT` + `io_uring` expert reads in place of Win32 `OVERLAPPED`;
3. `mmap`/`madvise` budgeted host backing and `/proc` memory telemetry.

All queue waits must remain finite; Linux support must not introduce persistent
GPU kernels, CPU/GPU busy polling, or an unbounded I/O queue. Conversion tools
are ordinary Python and can already run natively on Linux. Wine execution is
not a supported or benchmarked inference configuration.

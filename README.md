# UnbiasedStrata

A fork of [Strata](https://github.com/Niko1221/Strata) with the opposite priorities from upstream: instead of a
Python installer that downloads models and supports every card and OS, one C++ binary that serves one model family
(Qwen3.8-Flash-Next) through an OpenAI-style API. The pinned engine runs unmodified as a child process, exactly as
upstream tested it; everything around it — setup, server, installer — is replaced by a small C++ front end. Copy
the source, build it offline, start it. What was cut and why is under Compared with upstream Strata below.

[Strata](https://github.com/Niko1221/Strata)'s inference engine and llama.cpp's tokenizer, chat-template and
HTTP code, built into one binary that serves Qwen3.8-Flash-Next over an OpenAI-style API, on Linux or Windows.
No network is needed to build or run it: copy the source, run `make build` (Windows: `.\build.ps1`), start the
binary. No Python, no installer.

## Requirements

| | Linux | Windows |
| --- | --- | --- |
| Build | CMake 3.24+, a C++20 compiler (tested: g++ 13.3), CUDA toolkit 12.8+ (tested: 12.8; 13.0+ advised for RTX 50) | Visual Studio 2019 or 2022 with the C++ tools (tested: Build Tools 17.14), CMake 3.24+, CUDA toolkit 13.0 to 13.2 (tested: 13.0.3) |
| Run | the NVIDIA driver, `libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12` | the NVIDIA driver; `cublas64_13.dll` and `cublasLt64_13.dll`, which `build.ps1` puts beside the program |
| Network access | none, to build or to run | none, to build or to run |

Hardware: x86-64 with AVX2 and one NVIDIA card (RTX 30, 40 or 50). How much depends on the model:

| Model | Card | RAM | Checked on |
| --- | --- | --- | --- |
| Flash-Next IQ3_XXS | 24 GB or more | about 80 GB (the low-RAM mode, below about 53 GB, is untested) | Linux: RTX 3090 with 125 GB, RTX 4090 with 82 GB |
| Coder IQ1_M | 16 GB or more | 64 GB | Windows: RTX 5070 Ti |

Where RAM holds it (about 80 GB for IQ3_XXS, Linux only) the n-gram table is read into RAM at start, and a model on
a network volume costs start-up time only; Linux also retries the reads such a volume refuses for a moment. Elsewhere
the table is read from the disk as prompts need it, so keep the model on a local SSD (on Windows, always).

## Build

```
rsync -a UnbiasedStrata/ host:UnbiasedStrata/   # or: git clone, then git submodule update --init
ssh host 'cd UnbiasedStrata && make build'      # produces build/unbiased-strata
```

The tree is 6 MB plus the pinned llama.cpp submodule in `third_party/` (about 170 MB checked out). The
build fetches nothing. `make build JOBS=N` sets the job count; `CUDA_ARCHS` defaults to `86;89;120`.

## Run

```
./unbiased-strata --model /models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
                  --pack /models/pack-iq3-xxs --mtp /models/mtp
```

| | |
| --- | --- |
| Endpoints | `POST /v1/chat/completions` (streaming or not), `GET /v1/models`, `GET /health`, on `127.0.0.1:8080` |
| Options | `--ctx N` (32768), `--threads N` (default: physical cores, or half the CPU quota in a container), `--host`, `--port`, `--mmproj PATH`, `--vision-tokens N` (1024), `--ple-io`, `--low-ram` (both below), `--expert-profile PATH` (default: the pack's); engine flags after `--` |
| Memory | `--ple-io auto` (default) keeps the 28.8 GB n-gram table in RAM when about 80 GB of RAM holds it beside the 42.9 GB of experts, and otherwise reads it from the disk as needed (`direct`; keep the model on an SSD). `--low-ram auto` (default) turns on the low-RAM mode (`--resident-experts`: the GPU holds what it can, the rest is copied into RAM) below about 53 GB. Both choices are logged at start; `direct`, `mmap`, `ram`, `on` and `off` override them |
| Sampling | greedy unless `temperature` > 0 |
| Reasoning | returned as `reasoning_content`; `"reasoning_effort": "none"` turns it off |
| Tools | OpenAI format (`tools`, `tool_calls`) |
| Images | with `--mmproj`, OpenAI `image_url` parts holding a base64 `data:image/...` URL (PNG, JPEG, BMP or GIF), several per message |
| Not supported | more than one GPU, batch requests |

Images: `--mmproj /models/mmproj-Qwen3.8-Flash-Next-BF16.gguf` loads the model's own vision encoder (from the same
Hugging Face repository as the GGUF; 0.9 GB). The encoder runs on the CPU, so the card keeps its expert cache:
about 25 s and 1,000 prompt tokens for a 1-megapixel screenshot on 8 threads, 0.2 s for a small image; a repeated
image is served from a cache. `GET /v1/models` then lists `image` under `input_modalities`. Without `--mmproj` an
image request is refused with a 400.

The engine runs as a child process. When it exits or its stall watchdog fires, the front end exits with a failure
status, for a supervisor such as systemd (`Restart=on-failure`) to restart; `/health` and `/v1/models` answer 503
while the engine is down. Reasoning streams as it is generated; content and tool calls are sent once the whole
message is parsed, so a partial tool-call JSON never changes arguments already sent. `make test-frontend` checks
the child handling, the protocol and the memory and model rules without a model, and that `main.cpp` compiles;
`make test-chat-parser MODEL=<shard 1>` checks the chat template and tool-call parsing against the real GGUF. CI
(`.github/workflows/frontend-tests.yml`) runs the first on Linux and its Windows counterpart on every push; the
engine itself needs a GPU and is checked by hand.

## Windows

Checked on one machine: RTX 5070 Ti (16 GB), Ryzen 9 3900XT, 64 GB RAM,
Windows 11, Visual Studio 2022 Build Tools 17.14, CUDA 13.0.3, with the Coder IQ1_M model.

Build: Visual Studio 2019 or 2022 with the C++ tools (CUDA 13.0 to 13.2 accept only these), the CUDA 13 toolkit
(the compiler and the libraries; not the driver), CMake 3.24+ and the submodule. In PowerShell 7:

```
.\build.ps1          # produces build\win\unbiased-strata.exe, with cublas64_13.dll and cublasLt64_13.dll beside it
.\build.ps1 -Test    # also runs the front end's tests (.\build.ps1 -TestOnly needs no CUDA)
```

`-CudaArchs` defaults to `86;89;120`, as on Linux. The program needs only the NVIDIA driver and those two DLLs;
the CUDA and C++ runtimes are linked in. `cmake --build build\win --target chat-parser-test`, then
`build\win\chat-parser-test.exe <shard 1>`, checks the chat template and tool calls against the GGUF.

Run, with the same options as on Linux:

```
.\build\win\unbiased-strata.exe --model D:\models\coder-IQ1_M\Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf `
    --pack D:\packs\coder-iq1_m --mtp D:\mtp\rt --expert-profile D:\Strata\data\expert-profile-coder.bin
```

- `--ple-io ram` does not exist on Windows; `auto` reads the n-gram table from the disk (`direct`), so keep the
  model on an SSD.
- Speed on the machine above, Coder IQ1_M, through the server (mean of 8 prompts of 400 generated tokens; mean of 2
  prompts of ~19,500 tokens): 83.8 tokens/s generating and 2,903 tokens/s reading at 32K context; 82.6 and 2,812 at
  64K, where the KV cache past 32K positions is streamed from RAM (`--kv-resident 32768`, chosen automatically above
  32K when RAM allows). For the Coder the server also has its expert cache adapt every round (`--adapt-every 1
  --adapt-swaps 80 --adapt-decay 0.97`, upstream's tested values): 72.6 tokens/s without them.
- The engine runs in a Job Object: when the server ends, even by a crash or `Stop-Process -Force`, the engine
  ends with it and the GPU is freed.
- With `--mmproj`, image files go to an `images` folder beside the program, whose path must not contain a space
  (its 8.3 short form is used when it has one). Image input has not been tested on Windows yet.

Restarts: the server exits with a failure status when its engine fails, as on Linux. `.\serve.ps1` (same
arguments; engine options after a quoted `'--'`) starts it again after 10 s, and stops for good on a clean exit,
a bad option or a missing file or DLL. As a service (not tested yet), use [WinSW](https://github.com/winsw/winsw) with a
`unbiased-strata.xml` beside `WinSW-x64.exe` renamed to `unbiased-strata.exe`, then `unbiased-strata.exe install`:

```xml
<service>
  <id>unbiased-strata</id>
  <name>UnbiasedStrata</name>
  <executable>D:\UnbiasedStrata\build\win\unbiased-strata.exe</executable>
  <arguments>--model D:\models\...-00001-of-00002.gguf --pack D:\packs\... --mtp D:\mtp\rt</arguments>
  <onfailure action="restart" delay="10 sec"/>
  <stoptimeout>15 sec</stoptimeout>
  <log mode="roll-by-size"/>
</service>
```

## Supported models

Two releases of Qwen3.8-Flash-Next (mixture of experts). The server reads which one it is from the GGUF and
refuses any other model, quantization or pruned variant. The draft (MTP) head is required for both.

| Model | Source | Experts in RAM | Checked on |
| --- | --- | --- | --- |
| The full model (125B parameters) at IQ3_XXS | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), directory `IQ3_XXS`, revision `ed59f920` | 42.9 GB | the hosts under Status |
| The full model (125B parameters) at IQ3_S | the same repository, directory `IQ3_S`, revision `ed59f920` | 50.3 GB | not yet: needs a 64 GB PC with little else running |
| The Coder (256 of the 512 experts kept) at IQ1_M | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF), directory `IQ1_M`, revision `5348543e` | 23.4 GB | RTX 5070 Ti under Windows: chat, streaming and tool calls; images not yet |

The files below are the full model's at IQ3_XXS; for IQ3_S substitute shard 1 below (shard 2, the pack's
shape and the draft head are unchanged, but the pack itself is per-quantization: see UPSTREAM.md). The Coder's are the same set (shard 1 is 29.61 GB, shard 2 28.80 GB). Its
pack has no `expert-profile.bin`: upstream ships the Coder's profile as `data/expert-profile-coder.bin`, which
`--expert-profile PATH` points to.

Everything below must be on disk before start. Nothing is fetched.

| What | Size | SHA-256 or source |
| --- | --- | --- |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` (weights) | 47.04 GB | `219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15` |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` (n-gram table), beside shard 1 | 28.80 GB | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` (weights, instead of the IQ3_XXS shard 1) | 54.82 GB | `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3` |
| Pack: `index.txt`, `dense.bin`, `native_experts.txt`, `conversions.json`, `expert-profile.bin` | 1.5 GB | made once with upstream's tools, one pack per quantization (see UPSTREAM.md); not published yet |
| Draft head: `experts.bin`, `dense.bin`, `dense.txt`, `draft_vocab.bin` | 0.8 GB | made once with upstream's tools; not published yet |

About 78 GB (IQ3_XXS) or 86 GB (IQ3_S) in total. The commands that make the pack and draft head, and the SHA-256 of every file they
produce, are in [UPSTREAM.md](UPSTREAM.md).

## Status

Checked on one machine: RTX 3090, 32 vCPUs of an EPYC 7763, 125 GB RAM, CUDA 12.8, in a container.

| | |
| --- | --- |
| Build | from an rsynced tree with no git metadata; the binary holds no download code |
| Serving | chat, streaming, reasoning, tool calls; a coding agent (pi) wrote and ran a program through it |
| RTX 4090 (24 GB), 82 GB RAM, Debian, bare metal | serving with tool calls and images, through pi (file attachments and clipboard paste) and curl, over several days of use |
| Start from local disk | 19 s with the files in the OS file cache |
| Start from a network volume | 12 min for 76 GB (S3-backed RunPod volume, 120 to 330 MB/s). The volume refused reads twice with "cannot allocate memory"; the loader retried and went on |
| Not tested | RTX 50 cards; a cold start from local disk |

Speed, one run of each, same machine and model files, greedy sampling, reasoning off. Generation is the mean
of 8 different prompts of 400 tokens; prompt speed is the mean of 2 different prompts of about 20,000 tokens.

| | Generation, tokens/s | Prompt, tokens/s |
| --- | --- | --- |
| This fork | 107 (87 to 116) | 2,483 |
| Upstream, with `--pool-workers 15 --ple-io ram` | 78 (61 to 93) | 2,439 |
| Upstream, arguments as its installer writes them | 20 (16 to 21) | 2,368 |

Upstream as installed started 127 CPU workers on the container's 32 vCPUs and read the n-gram table from
disk per token; the middle row removes both. The remaining gap to the fork is not explained yet: the two
builds differ in compile flags and in how the n-gram table is held in RAM, and each row is a single run.

## Known problems

- Text in a message that looks like a control token (`<think>`, `<tool_call>`) is tokenized as that control
  token. Upstream's tokenizer keeps it as plain text.
- Replies are not reproducible token for token between runs or between this fork and upstream, even with
  greedy sampling, so "same output" cannot be checked by comparing text.

## Compared with upstream Strata (6f32ec0)

| | Upstream | This fork |
| --- | --- | --- |
| Engine source (`src/`, `include/`) | 272 files | two files changed, one added (retrying reads from slow stores); cutting it down is the next work |
| Build | clones llama.cpp during CMake | pinned submodule, fetches nothing; one patch, applied by `make build`: tool-call arguments in any order (see UPSTREAM.md) |
| Setup | Python installer downloads the model and builds the pack and draft head | none; files must already exist |
| Server | Python | C++, in the same binary (`frontend/`), with the image encoder from llama.cpp's `mtmd` |
| Python | 33,745 lines | 0 |
| C, C++ and CUDA | 208,254 lines (with tests and the Intel port) | 84,086 lines |
| Tracked files | 851, 27 MB | 292, 6 MB |
| Configurations | several models, NVIDIA, AMD and Intel, Linux and Windows, multi-GPU | two models, one NVIDIA card, Linux and Windows |

## Credits and licence

MIT, as upstream. The engine is the work of Niko1221 and the Strata contributors; see [LICENSE](LICENSE) and
[UPSTREAM.md](UPSTREAM.md). llama.cpp and ggml are MIT. The model files have their own licences.

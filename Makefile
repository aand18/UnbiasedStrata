# UnbiasedStrata build.  Needs CMake 3.24+, a C++20 compiler and the CUDA toolkit.  Nothing is downloaded.
BUILD_DIR  ?= build
# RTX 30 (86), RTX 40 (89) and RTX 50 (120) in one binary.  sm_120 needs CUDA 12.8+, and upstream recommends 13.0+.
CUDA_ARCHS ?= 86;89;120
# capped: inside a container nproc reports the host's CPUs, not the container's quota
JOBS       ?= $(shell n=$$(nproc 2>/dev/null || echo 4); [ $$n -gt 16 ] && n=16; echo $$n)

.PHONY: build clean

build:
	@test -f third_party/llama.cpp/ggml/CMakeLists.txt || \
	  { echo "third_party/llama.cpp is empty: run 'git submodule update --init' first"; exit 1; }
	@for p in ../../patches/llama.cpp-qwen3-coder-any-argument-order.patch ../../patches/llama.cpp-clip-mapped-weights.patch; do \
	  git -C third_party/llama.cpp apply --check --reverse $$p 2>/dev/null || \
	  git -C third_party/llama.cpp apply $$p; done
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON \
	      -DSTRATA_PORTABLE=ON -DSTRATA_BUILD_TESTS=OFF "-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCHS)"
	cmake --build $(BUILD_DIR) -j $(JOBS) --target unbiased-strata
	@echo "built: $(BUILD_DIR)/unbiased-strata"

clean:
	rm -rf $(BUILD_DIR)

# Protocol and child supervision regressions; no CUDA/model required.
.PHONY: test-frontend
test-frontend:
	mkdir -p $(BUILD_DIR)/frontend-tests
	$(CXX) -std=c++20 -Wall -Wextra -Werror frontend/child_watch_test.cpp -o $(BUILD_DIR)/frontend-tests/child-watch-test
	$(BUILD_DIR)/frontend-tests/child-watch-test
	$(CXX) -std=c++20 -Wall -Wextra -Werror frontend/engine_test.cpp -o $(BUILD_DIR)/frontend-tests/engine-test
	$(BUILD_DIR)/frontend-tests/engine-test
	$(CXX) -std=c++20 -Wall -Wextra -Werror -pthread frontend/engine_backpressure_test.cpp -o $(BUILD_DIR)/frontend-tests/engine-backpressure-test
	$(BUILD_DIR)/frontend-tests/engine-backpressure-test
	$(CXX) -std=c++20 -Wall -Wextra -Werror frontend/process_test.cpp -o $(BUILD_DIR)/frontend-tests/process-test
	$(BUILD_DIR)/frontend-tests/process-test
	$(CXX) -std=c++20 -Wall -Wextra -Werror frontend/memory_plan_test.cpp -o $(BUILD_DIR)/frontend-tests/memory-plan-test
	$(BUILD_DIR)/frontend-tests/memory-plan-test
	@# main.cpp compiles (syntax only: no CUDA, no libraries), when the llama.cpp headers are there
	@if [ -f third_party/llama.cpp/include/llama.h ]; then \
	  $(CXX) -std=c++20 -Wall -Wextra -Werror -fsyntax-only -isystem third_party/llama.cpp/include \
	    -isystem third_party/llama.cpp/common -isystem third_party/llama.cpp/ggml/include \
	    -isystem third_party/llama.cpp/vendor -isystem third_party/llama.cpp/tools/mtmd frontend/main.cpp && \
	  echo "frontend/main.cpp compiles (-Werror)"; fi

# Model-template compatibility; CPU only, after the regular build produced llama libraries.
.PHONY: test-chat-parser
test-chat-parser:
	@test -n "$(MODEL)" || { echo 'set MODEL to the first GGUF shard'; exit 2; }
	mkdir -p $(BUILD_DIR)/frontend-tests
	$(CXX) -std=c++20 -O0 -Ifrontend -Ithird_party/llama.cpp/include -Ithird_party/llama.cpp/common \
	  -Ithird_party/llama.cpp/ggml/include -Ithird_party/llama.cpp/vendor -Ithird_party/llama.cpp/tools/mtmd \
	  frontend/chat_parser_test.cpp \
	  -o $(BUILD_DIR)/frontend-tests/chat-parser-test $(BUILD_DIR)/llama.cpp/tools/mtmd/libmtmd.a \
	  $(BUILD_DIR)/llama.cpp/vendor/hash/libvendor-hash.a $(BUILD_DIR)/llama.cpp/common/libllama-common.a \
	  $(BUILD_DIR)/llama.cpp/src/libllama.a $(BUILD_DIR)/llama.cpp/vendor/cpp-httplib/libcpp-httplib.a \
	  $(BUILD_DIR)/ggml/src/libggml.a $(BUILD_DIR)/ggml/src/libggml-cpu.a $(BUILD_DIR)/ggml/src/libggml-base.a \
	  $(BUILD_DIR)/llama.cpp/common/libllama-common-base.a -lpthread -ldl -lm
	$(BUILD_DIR)/frontend-tests/chat-parser-test "$(MODEL)"

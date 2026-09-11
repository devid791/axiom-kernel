PREFIX ?= /usr/local
.DEFAULT_GOAL := all
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CXX ?= c++
CUDA_ARCH ?= sm_120

BUILD_DIR ?= build
LIB_DIR ?= lib
BIN_DIR ?= bin

VISION ?= 1
MEDIA ?= 0

CPPFLAGS += -Iinclude -I$(CUDA_HOME)/include
CXXFLAGS ?= -O3 -Wall -Wextra -std=c++17 -fPIC
NVCCFLAGS ?= -O3 --use_fast_math -std=c++17 -arch=$(CUDA_ARCH) -Xcompiler -fPIC
CUDA_LIBS ?= -lcudart -lcublasLt -lcublas -ldl -lpthread -lm -lcrypto
CUDA_LDFLAGS ?= -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/targets/x86_64-linux/lib
VISION_LIBS ?= -ljpeg -lpng16 -lwebp
MEDIA_LIBS ?= -lavformat -lavcodec -lavutil -lswscale

FLASHINFER_CPPFLAGS := -Ithird_party/flashinfer/include

CORE_CPP_SOURCES := \
    src/axiom_runtime.cpp \
    src/axiom_qwen38_dspark.cpp \
    src/axiom_qwen38_fp8_bank.cpp \
    src/axiom_qwen38_kv_tier.cpp \
    src/axiom_qwen38_mlp_bank.cpp \
    src/axiom_qwen38_mtp.cpp \
    src/axiom_qwen38_mtp_speculative.cpp \
    src/axiom_qwen38_nvfp4_bank.cpp \
    src/axiom_qwen38_session_store.cpp \
    src/axiom_qwen38_speculative.cpp \
    src/axiom_qwen38_swarm_scheduler.cpp \
    src/axiom_sha256.cpp

CORE_CU_SOURCES := \
    src/axiom_cuda.cu \
    src/axiom_cuda_precise.cu \
    src/axiom_cuda_nvfp4.cu \
    src/axiom_cuda_nvfp4_moe.cu \
    src/axiom_qwen38_attention.cu \
    src/axiom_qwen38_bf16_linear.cu \
    src/axiom_qwen38_dspark_compute.cu \
    src/axiom_qwen38_flashinfer.cu \
    src/axiom_qwen38_fp8.cu \
    src/axiom_qwen38_fp8_mlp.cu \
    src/axiom_qwen38_gdn.cu \
    src/axiom_qwen38_model.cu \
    src/axiom_qwen38_mtp_compute.cu \
    src/axiom_qwen38_mtp_device_control.cu \
    src/axiom_qwen38_nvfp4.cu \
    src/axiom_qwen38_nvfp4_mlp.cu \
    src/axiom_qwen38_vision.cu

ifeq ($(VISION),1)
CORE_CPP_SOURCES += src/axiom_qwen38_vision_preprocess.cpp
VISION_LINK_LIBS := $(VISION_LIBS)
endif
ifeq ($(MEDIA),1)
CORE_CPP_SOURCES += src/axiom_qwen38_video_decode.cpp
MEDIA_LINK_LIBS := $(MEDIA_LIBS)
endif

CORE_CPP_OBJECTS := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(CORE_CPP_SOURCES))
CORE_CU_OBJECTS := $(patsubst src/%.cu,$(BUILD_DIR)/%.o,$(CORE_CU_SOURCES))
CORE_OBJECTS := $(CORE_CPP_OBJECTS) $(CORE_CU_OBJECTS)

# Preserve the native reference math contract; global fast-math changes
# RoPE/normalization and can change exact speculative acceptance.
$(filter $(BUILD_DIR)/axiom_qwen38_%,$(CORE_CU_OBJECTS)) $(BUILD_DIR)/axiom_cuda_precise.o: NVCCFLAGS := $(filter-out --use_fast_math,$(NVCCFLAGS))

COMMON_LINK_LIBS := $(CUDA_LDFLAGS) $(CUDA_LIBS) $(VISION_LINK_LIBS) $(MEDIA_LINK_LIBS)

.PHONY: all clean check public-scan host-tests smoke qwen38-generate qwen38-temporal-gate qwen38-speculative-graph-gate qwen38-paged-runtime-gate qwen38-dspark-abi-gate install cuda-check

all: cuda-check $(LIB_DIR)/libaxiom.so $(BIN_DIR)/axiom-qwen38-generate \
     $(BIN_DIR)/axiom-qwen38-dspark-abi-gate host-tests

cuda-check:
	@test -x "$(NVCC)" || { echo "Axiom CUDA build requires nvcc at $(NVCC); host-only tests remain available via make host-tests" >&2; exit 2; }

$(BUILD_DIR) $(BIN_DIR) $(LIB_DIR):
	mkdir -p $@

$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/%.o: src/%.cu | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(FLASHINFER_CPPFLAGS) $(NVCCFLAGS) -MMD -MP -c $< -o $@

-include $(wildcard $(BUILD_DIR)/*.d)

$(LIB_DIR)/libaxiom.so: $(CORE_OBJECTS) | $(LIB_DIR) cuda-check
	$(NVCC) -shared -o $@ $^ $(COMMON_LINK_LIBS)

$(BUILD_DIR)/axiom_qwen38_generate.o: tools/axiom_qwen38_generate.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR)/axiom-qwen38-generate: $(BUILD_DIR)/axiom_qwen38_generate.o $(LIB_DIR)/libaxiom.so | $(BIN_DIR)
	$(CXX) -o $@ $< -L$(LIB_DIR) -laxiom $(COMMON_LINK_LIBS) -Wl,-rpath,'$$ORIGIN/../lib'

qwen38-generate: $(BIN_DIR)/axiom-qwen38-generate

$(BUILD_DIR)/axiom_qwen38_temporal_gate.o: tools/axiom_qwen38_temporal_gate.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR)/axiom-qwen38-temporal-gate: $(BUILD_DIR)/axiom_qwen38_temporal_gate.o $(LIB_DIR)/libaxiom.so | $(BIN_DIR)
	$(CXX) -o $@ $< -L$(LIB_DIR) -laxiom $(COMMON_LINK_LIBS) -Wl,-rpath,'$$ORIGIN/../lib'

qwen38-temporal-gate: $(BIN_DIR)/axiom-qwen38-temporal-gate

$(BUILD_DIR)/axiom_qwen38_speculative_graph_gate.o: tools/axiom_qwen38_speculative_graph_gate.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR)/axiom-qwen38-speculative-graph-gate: $(BUILD_DIR)/axiom_qwen38_speculative_graph_gate.o $(LIB_DIR)/libaxiom.so | $(BIN_DIR)
	$(CXX) -o $@ $< -L$(LIB_DIR) -laxiom $(COMMON_LINK_LIBS) -Wl,-rpath,'$$ORIGIN/../lib'

qwen38-speculative-graph-gate: $(BIN_DIR)/axiom-qwen38-speculative-graph-gate

$(BUILD_DIR)/axiom_qwen38_paged_runtime_gate.o: tools/axiom_qwen38_paged_runtime_gate.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR)/axiom-qwen38-paged-runtime-gate: $(BUILD_DIR)/axiom_qwen38_paged_runtime_gate.o $(LIB_DIR)/libaxiom.so | $(BIN_DIR)
	$(CXX) -o $@ $< -L$(LIB_DIR) -laxiom $(COMMON_LINK_LIBS) -Wl,-rpath,'$$ORIGIN/../lib'

qwen38-paged-runtime-gate: $(BIN_DIR)/axiom-qwen38-paged-runtime-gate

$(BUILD_DIR)/axiom_qwen38_dspark_abi_gate.o: tests/axiom_qwen38_dspark_abi_gate.cpp include/axiom/axiom.h include/axiom/qwen38_dspark_compute.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR)/axiom-qwen38-dspark-abi-gate: $(BUILD_DIR)/axiom_qwen38_dspark_abi_gate.o $(LIB_DIR)/libaxiom.so | $(BIN_DIR)
	$(CXX) -o $@ $< -L$(LIB_DIR) -laxiom $(COMMON_LINK_LIBS) -Wl,-rpath,'$$ORIGIN/../lib'

qwen38-dspark-abi-gate: $(BIN_DIR)/axiom-qwen38-dspark-abi-gate
	$(BIN_DIR)/axiom-qwen38-dspark-abi-gate

$(BUILD_DIR)/axiom_qwen38_session_store_test.o: tests/axiom_qwen38_session_store_test.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/axiom_qwen38_swarm_scheduler_test.o: tests/axiom_qwen38_swarm_scheduler_test.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR)/axiom-qwen38-session-store-test: $(BUILD_DIR)/axiom_qwen38_session_store_test.o $(BUILD_DIR)/axiom_qwen38_session_store.o | $(BIN_DIR)
	$(CXX) -o $@ $^ -lcrypto

$(BIN_DIR)/axiom-qwen38-swarm-scheduler-test: $(BUILD_DIR)/axiom_qwen38_swarm_scheduler_test.o $(BUILD_DIR)/axiom_qwen38_swarm_scheduler.o | $(BIN_DIR)
	$(CXX) -o $@ $^

host-tests: $(BIN_DIR)/axiom-qwen38-session-store-test $(BIN_DIR)/axiom-qwen38-swarm-scheduler-test
	$(BIN_DIR)/axiom-qwen38-session-store-test
	$(BIN_DIR)/axiom-qwen38-swarm-scheduler-test

$(BUILD_DIR)/axiom_smoke.o: tests/axiom_smoke.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/axiom_model_smoke.o: tests/axiom_model_smoke.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/axiom_nvfp4_smoke.o: tests/axiom_nvfp4_smoke.cpp | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/axiom_quant_smoke.o: tests/axiom_quant_smoke.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/axiom_safetensors_smoke.o: tests/axiom_safetensors_smoke.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/axiom_tokenizer_smoke.o: tests/axiom_tokenizer_smoke.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR)/axiom-smoke: $(BUILD_DIR)/axiom_smoke.o $(LIB_DIR)/libaxiom.so | $(BIN_DIR)
	$(CXX) -o $@ $< -L$(LIB_DIR) -laxiom $(COMMON_LINK_LIBS) -Wl,-rpath,'$$ORIGIN/../lib'
$(BIN_DIR)/axiom-model-smoke: $(BUILD_DIR)/axiom_model_smoke.o $(LIB_DIR)/libaxiom.so | $(BIN_DIR)
	$(CXX) -o $@ $< -L$(LIB_DIR) -laxiom $(COMMON_LINK_LIBS) -Wl,-rpath,'$$ORIGIN/../lib'
$(BIN_DIR)/axiom-nvfp4-smoke: $(BUILD_DIR)/axiom_nvfp4_smoke.o $(LIB_DIR)/libaxiom.so | $(BIN_DIR)
	$(NVCC) -o $@ $< -L$(LIB_DIR) -laxiom $(COMMON_LINK_LIBS) -Wl,-rpath,'$$ORIGIN/../lib'
$(BIN_DIR)/axiom-quant-smoke: $(BUILD_DIR)/axiom_quant_smoke.o $(LIB_DIR)/libaxiom.so | $(BIN_DIR)
	$(CXX) -o $@ $< -L$(LIB_DIR) -laxiom $(COMMON_LINK_LIBS) -Wl,-rpath,'$$ORIGIN/../lib'
$(BIN_DIR)/axiom-safetensors-smoke: $(BUILD_DIR)/axiom_safetensors_smoke.o $(LIB_DIR)/libaxiom.so | $(BIN_DIR)
	$(CXX) -o $@ $< -L$(LIB_DIR) -laxiom $(COMMON_LINK_LIBS) -Wl,-rpath,'$$ORIGIN/../lib'
$(BIN_DIR)/axiom-tokenizer-smoke: $(BUILD_DIR)/axiom_tokenizer_smoke.o $(LIB_DIR)/libaxiom.so | $(BIN_DIR)
	$(CXX) -o $@ $< -L$(LIB_DIR) -laxiom $(COMMON_LINK_LIBS) -Wl,-rpath,'$$ORIGIN/../lib'

smoke: $(BIN_DIR)/axiom-smoke $(BIN_DIR)/axiom-model-smoke $(BIN_DIR)/axiom-nvfp4-smoke \
       $(BIN_DIR)/axiom-quant-smoke $(BIN_DIR)/axiom-safetensors-smoke $(BIN_DIR)/axiom-tokenizer-smoke
	$(BIN_DIR)/axiom-smoke
	$(BIN_DIR)/axiom-model-smoke
	$(BIN_DIR)/axiom-nvfp4-smoke
	$(BIN_DIR)/axiom-quant-smoke
	$(BIN_DIR)/axiom-safetensors-smoke
	$(BIN_DIR)/axiom-tokenizer-smoke

check: public-scan host-tests

public-scan:
	./scripts/public_scan.sh

install: all
	install -d $(DESTDIR)$(PREFIX)/lib $(DESTDIR)$(PREFIX)/include/axiom
	install -m 0644 $(LIB_DIR)/libaxiom.so $(DESTDIR)$(PREFIX)/lib/
	cp -R include/axiom/. $(DESTDIR)$(PREFIX)/include/axiom/

clean:
	$(RM) -r $(BUILD_DIR) $(BIN_DIR) $(LIB_DIR)

include mk/public_extensions.mk
include mk/codex_provider.mk

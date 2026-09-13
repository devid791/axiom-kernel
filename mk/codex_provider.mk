# Opt-in native HTTP/Responses integration. These targets never install a service
# or load weights; model-backed qualification is a separate explicit operation.
.PHONY: qwen38-api codex-provider-test codex-provider-api-test vision-memory-budget-test codex-host-tests codex-media-check

codex-media-check:
	@test "$(MEDIA)" = 1 -a "$(VISION)" = 1 || { echo "The HTTP provider requires MEDIA=1 VISION=1 and FFmpeg/image development libraries; the standalone kernel remains available without MEDIA." >&2; exit 2; }

$(BUILD_DIR)/axiom_qwen38_api.o: tools/axiom_qwen38_api.cpp tools/axiom_codex_provider.h tools/axiom_codex_schema.h tools/axiom_codex_prefill_graph.h tools/axiom_aliced_json.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BIN_DIR)/axiom-qwen38-api: $(BUILD_DIR)/axiom_qwen38_api.o $(BUILD_DIR)/axiom_qwen38_spec_identity_lib.o $(LIB_DIR)/libaxiom.so | $(BIN_DIR) codex-media-check
	$(CXX) $(CXXFLAGS) -o $@ $(BUILD_DIR)/axiom_qwen38_api.o $(BUILD_DIR)/axiom_qwen38_spec_identity_lib.o -L$(LIB_DIR) -laxiom $(COMMON_LINK_LIBS) -Wl,-rpath,'$$ORIGIN/../lib'

qwen38-api: $(BIN_DIR)/axiom-qwen38-api

$(BIN_DIR)/axiom-codex-provider-test: tests/axiom_codex_provider_test.cpp tools/axiom_codex_provider.h tools/axiom_codex_schema.h tools/axiom_aliced_json.h | $(BIN_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -Itools $< -lcrypto -o $@

codex-provider-test: $(BIN_DIR)/axiom-codex-provider-test
	$<

$(BIN_DIR)/axiom-codex-provider-api-test: tests/axiom_codex_provider_api_test.cpp tools/axiom_qwen38_api.cpp tools/axiom_codex_provider.h tools/axiom_codex_schema.h tools/axiom_codex_prefill_graph.h tools/axiom_aliced_json.h $(BUILD_DIR)/axiom_qwen38_spec_identity_lib.o $(LIB_DIR)/libaxiom.so | $(BIN_DIR) codex-media-check
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -Itools $< $(BUILD_DIR)/axiom_qwen38_spec_identity_lib.o -L$(LIB_DIR) -laxiom $(COMMON_LINK_LIBS) -Wl,-rpath,'$$ORIGIN/../lib' -o $@

codex-provider-api-test: $(BIN_DIR)/axiom-codex-provider-api-test
	$<

$(BIN_DIR)/vision-memory-budget-test: tests/vision_memory_budget_test.cpp include/axiom/vision_memory_budget.hpp | $(BIN_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

vision-memory-budget-test: $(BIN_DIR)/vision-memory-budget-test
	$<

codex-host-tests: codex-provider-test codex-provider-api-test vision-memory-budget-test qwen38-api
	$(BIN_DIR)/axiom-qwen38-api --self-test-tool-contract
	$(BIN_DIR)/axiom-qwen38-api --self-test-stream-progress
	$(BIN_DIR)/axiom-qwen38-api --self-test-utf8
	$(BIN_DIR)/axiom-qwen38-api --self-test-speculative-routing

# Conversation continuity gates are model-free unless a tokenizer/live endpoint
# is explicitly supplied. No test target starts a production service.
$(BUILD_DIR)/axiom_qwen38_api.o $(BIN_DIR)/axiom-codex-provider-api-test: include/axiom/qwen38_prefix_reuse.hpp include/axiom/qwen38_request_progress.hpp
$(BUILD_DIR)/axiom_qwen38_attention.o: include/axiom/qwen38_mixed_kv.hpp

$(BIN_DIR)/axiom-qwen38-prefix-reuse-test: tests/axiom_qwen38_prefix_reuse_test.cpp include/axiom/qwen38_prefix_reuse.hpp | $(BIN_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

$(BIN_DIR)/axiom-qwen38-request-progress-test: tests/axiom_qwen38_request_progress_test.cpp include/axiom/qwen38_request_progress.hpp | $(BIN_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -pthread -o $@

$(BIN_DIR)/axiom-qwen38-mixed-kv-test: tests/axiom_qwen38_mixed_kv_test.cpp include/axiom/qwen38_mixed_kv.hpp | $(BIN_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

.PHONY: qwen38-prefix-reuse-test qwen38-request-progress-test qwen38-mixed-kv-test
qwen38-prefix-reuse-test: $(BIN_DIR)/axiom-qwen38-prefix-reuse-test
	$<
qwen38-request-progress-test: $(BIN_DIR)/axiom-qwen38-request-progress-test
	$<
qwen38-mixed-kv-test: $(BIN_DIR)/axiom-qwen38-mixed-kv-test
	$<

codex-host-tests: qwen38-prefix-reuse-test qwen38-request-progress-test qwen38-mixed-kv-test

$(BIN_DIR)/axiom-codex-prefix-replay-test: tests/axiom_codex_prefix_replay_test.cpp tools/axiom_qwen38_api.cpp tools/axiom_codex_provider.h include/axiom/qwen38_prefix_reuse.hpp include/axiom/qwen38_request_progress.hpp $(BUILD_DIR)/axiom_qwen38_spec_identity_lib.o $(LIB_DIR)/libaxiom.so | $(BIN_DIR) codex-media-check
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -Itools $< $(BUILD_DIR)/axiom_qwen38_spec_identity_lib.o -L$(LIB_DIR) -laxiom $(COMMON_LINK_LIBS) -Wl,-rpath,'$$ORIGIN/../lib' -o $@

.PHONY: codex-prefix-replay-test
codex-prefix-replay-test: $(BIN_DIR)/axiom-codex-prefix-replay-test
	@test -n "$(QWEN38_TOKENIZER_PATH)" || { echo 'Set QWEN38_TOKENIZER_PATH'; exit 2; }
	$< "$(QWEN38_TOKENIZER_PATH)"

$(BIN_DIR)/axiom-qwen38-mixed-kv-microbench: tools/axiom_qwen38_mixed_kv_microbench.cu include/axiom/qwen38_mixed_kv.hpp $(LIB_DIR)/libaxiom.so | $(BIN_DIR)
	$(NVCC) $(CPPFLAGS) $(filter-out --use_fast_math,$(NVCCFLAGS)) $< -L$(LIB_DIR) -laxiom $(COMMON_LINK_LIBS) -Xlinker -rpath -Xlinker '$$ORIGIN/../lib' -o $@

.PHONY: qwen38-mixed-kv-microbench
qwen38-mixed-kv-microbench: $(BIN_DIR)/axiom-qwen38-mixed-kv-microbench
	$<

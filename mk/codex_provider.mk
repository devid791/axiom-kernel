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

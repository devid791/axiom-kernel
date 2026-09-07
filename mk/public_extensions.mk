# Public build-only additions. No services, model downloads or deployment.
.PHONY: extended-host-tests qwen4exp-build qwen4exp-host-tests qwen4exp-text-model-build

$(BUILD_DIR)/axiom_sha256_test.o: tests/axiom_sha256_test.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BIN_DIR)/axiom-sha256-test: $(BUILD_DIR)/axiom_sha256_test.o $(BUILD_DIR)/axiom_sha256.o | $(BIN_DIR)
	$(CXX) -o $@ $^

$(BUILD_DIR)/axiom_qwen38_spec_identity_lib.o: tools/axiom_qwen38_spec_identity.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -Itools -MMD -MP -DAXIOM_QWEN38_SPEC_IDENTITY_LIBRARY_ONLY -c $< -o $@
$(BUILD_DIR)/axiom_qwen38_spec_identity_test.o: tests/axiom_qwen38_spec_identity_test.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BIN_DIR)/axiom-qwen38-spec-identity-test: $(BUILD_DIR)/axiom_qwen38_spec_identity_test.o $(BUILD_DIR)/axiom_qwen38_spec_identity_lib.o $(BUILD_DIR)/axiom_sha256.o | $(BIN_DIR)
	$(CXX) -o $@ $^

$(BUILD_DIR)/axiom_qwen38_kv_tier_transaction_test.o: tests/axiom_qwen38_kv_tier_transaction_test.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
$(BIN_DIR)/axiom-qwen38-kv-tier-transaction-test: $(BUILD_DIR)/axiom_qwen38_kv_tier_transaction_test.o $(LIB_DIR)/libaxiom.so | $(BIN_DIR)
	$(CXX) -o $@ $< -L$(LIB_DIR) -laxiom $(COMMON_LINK_LIBS) -Wl,-rpath,'$$ORIGIN/../lib'
extended-host-tests: host-tests $(BIN_DIR)/axiom-sha256-test $(BIN_DIR)/axiom-qwen38-spec-identity-test $(BIN_DIR)/axiom-qwen38-kv-tier-transaction-test
	$(BIN_DIR)/axiom-sha256-test
	$(BIN_DIR)/axiom-qwen38-spec-identity-test
	$(BIN_DIR)/axiom-qwen38-kv-tier-transaction-test

Q4_CPP := $(wildcard src/qwen4exp/*.cpp) src/axiom_qwen4exp_admission.cpp
Q4_CU := $(wildcard src/qwen4exp/*.cu)
Q4_OBJECTS := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(Q4_CPP)) $(patsubst src/%.cu,$(BUILD_DIR)/%.o,$(Q4_CU))
Q4_NVCCFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH) -Xcompiler -fPIC
$(BUILD_DIR)/qwen4exp:
	mkdir -p $@
$(BUILD_DIR)/qwen4exp/%.o: src/qwen4exp/%.cpp | $(BUILD_DIR)/qwen4exp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -Itools -MMD -MP -c $< -o $@
$(BUILD_DIR)/qwen4exp/%.o: src/qwen4exp/%.cu | $(BUILD_DIR)/qwen4exp
	$(NVCC) $(CPPFLAGS) $(Q4_NVCCFLAGS) -MMD -MP -c $< -o $@
$(BUILD_DIR)/axiom_qwen4exp_admission.o: src/axiom_qwen4exp_admission.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -Itools -MMD -MP -c $< -o $@
-include $(wildcard $(BUILD_DIR)/qwen4exp/*.d)
$(LIB_DIR)/libaxiom-qwen4exp.so: $(Q4_OBJECTS) $(LIB_DIR)/libaxiom.so
	$(NVCC) -shared -o $@ $(Q4_OBJECTS) -L$(LIB_DIR) -laxiom $(COMMON_LINK_LIBS) -lcuda -Xlinker --no-undefined -Xlinker -rpath -Xlinker '$$ORIGIN'
qwen4exp-build: $(LIB_DIR)/libaxiom-qwen4exp.so

$(BUILD_DIR)/qwen4exp/test_%.o: tests/qwen4exp/%_test.cu | $(BUILD_DIR)/qwen4exp
	$(NVCC) $(CPPFLAGS) $(Q4_NVCCFLAGS) -c $< -o $@
$(BUILD_DIR)/qwen4exp/test_%.o: tests/qwen4exp/%_test.cpp | $(BUILD_DIR)/qwen4exp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -Itools -c $< -o $@
$(BIN_DIR)/qwen4exp-text-model-test: $(BUILD_DIR)/qwen4exp/test_text_model.o $(LIB_DIR)/libaxiom-qwen4exp.so | $(BIN_DIR)
	$(NVCC) -o $@ $< -L$(LIB_DIR) -laxiom-qwen4exp -laxiom $(COMMON_LINK_LIBS) -lcuda -Xlinker -rpath -Xlinker '$$ORIGIN/../lib'
qwen4exp-text-model-build: $(BIN_DIR)/qwen4exp-text-model-test
$(BIN_DIR)/qwen4exp-ple-test: $(BUILD_DIR)/qwen4exp/test_ple.o $(BUILD_DIR)/qwen4exp/ple.o | $(BIN_DIR)
	$(CXX) -o $@ $^
$(BIN_DIR)/qwen4exp-expert-pager-test: $(BUILD_DIR)/qwen4exp/test_expert_pager.o $(BUILD_DIR)/qwen4exp/expert_pager.o | $(BIN_DIR)
	$(CXX) -o $@ $^ -pthread
qwen4exp-host-tests: $(BIN_DIR)/qwen4exp-ple-test $(BIN_DIR)/qwen4exp-expert-pager-test
	$(BIN_DIR)/qwen4exp-ple-test
	$(BIN_DIR)/qwen4exp-expert-pager-test

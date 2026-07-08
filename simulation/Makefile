MKDIR := mkdir -p
RM    := rm -f
RMDIR := rm -fr


SOURCES_DIR := ./p4sources
INCLUDE_DIR := ./p4include
OBJECTS_DIR := ./p4objects

PCAP_DIR := ./pcaps
LOG_DIR := ./logs
LOG_LEVEL := INFO
RUN_DIRS := $(PCAP_DIR) $(LOG_DIR)



PARTITIONING_TYPE ?= ofsp
GENERATED_P4_DIR := $(SOURCES_DIR)/p4_$(PARTITIONING_TYPE)
BUILTIN_P4_DIR := $(SOURCES_DIR)/built_in_p4
# dune_*.p4 in the generated dir are shared includes (headers/inference/ingress
# emitted for ensemble mode), not standalone programs, so they are not compiled.
SOURCES := $(filter-out $(GENERATED_P4_DIR)/dune_%.p4,$(wildcard $(GENERATED_P4_DIR)/*.p4)) $(wildcard $(BUILTIN_P4_DIR)/*.p4)
OBJECTS := $(foreach s,$(SOURCES),$(OBJECTS_DIR)/$(notdir $(s:.p4=.json)))
DEPS := $(foreach s,$(SOURCES),$(OBJECTS_DIR)/$(notdir $(s:.p4=.d)))

vpath %.p4 $(GENERATED_P4_DIR) $(BUILTIN_P4_DIR)
COMBINED_PCAPS_FILE := ./pcaps/combined.csv
GROUND_TRUTH_FILE ?= ./data/TON-IOT_test_flow_length.csv



P4C := p4c-bm2-ss

P4_FLAGS := --std p4-16 -I $(INCLUDE_DIR)
P4_FLAGS_RUNTIME = --p4runtime-files $(OBJECTS_DIR)/$*.p4.p4info.txtpb
P4_FLAGS_DEPS = -MD -MP -MT $(OBJECTS_DIR)/$*.json -MF $(OBJECTS_DIR)/$*.d



MN := sudo PATH="$(PATH)" `which mn`

MN_DIR := ./dune


join_with_comma = $(shell echo $1 | sed 's/ \+/,/g')


MN_CUSTOM_CLASSES := $(shell find $(MN_DIR) -type f -name '*.py')
MN_CUSTOM := --custom=$(call join_with_comma,$(MN_CUSTOM_CLASSES))


CUSTOM_TOPOLOGY := configs/topos/simpletopo.json
FATTREE_TOPOLOGY := configs/topos/fattreetopo.json
MODELS := configs/models/ton.json
MODELS_DIR := ./models
PCAP_REGEX ?= ^pe_l[0-9]+$
SUPER_SPINES ?= 4
SPINES ?= 4
LEAFS ?= 6
PODS ?= 4
HOSTS_PER_LEAF ?= 2
TEST_PPS ?= 3000
TEST_PCAP ?= ./data/ToN_IoT_test.pcap
TEST_PCAP_DIR ?= ./utils/experiment_pcaps
PKT_NUM ?=
RESULTS_BASE := ./results/$(PARTITIONING_TYPE)
RUN_ID := $(shell date +%Y%m%d_%H%M%S)
RESULTS_DIR = $(RESULTS_BASE)/run_$(RUN_ID)
RESULTS_FILE = $(RESULTS_DIR)/results_p$(PODS)_ss$(SUPER_SPINES)_s$(SPINES)_l$(LEAFS)_h$(HOSTS_PER_LEAF)_$(TEST_PPS)pps.txt
SAVE_RESULTS ?= NO

TOPO_ARGS = models=$(MODELS) \
			 models_dir=$(MODELS_DIR) \
			 objects_dir=$(OBJECTS_DIR) \
			 log_dir=$(LOG_DIR) \
			 pcap_regex=$(PCAP_REGEX) \
			 pcap_dir=$(PCAP_DIR) \
			 test_pps=$(TEST_PPS) \
			 pkt_num=$(PKT_NUM) \
			 super_spines=$(SUPER_SPINES) \
			 pods=$(PODS) \
			 spines=$(SPINES) \
			 leafs=$(LEAFS) \
			 hosts_per_leaf=$(HOSTS_PER_LEAF)

TOPO_ARGS_LINEAR = models=$(MODELS) \
			 models_dir=$(MODELS_DIR) \
			 objects_dir=$(OBJECTS_DIR) \
			 log_dir=$(LOG_DIR) \
			 pcap_dir=$(PCAP_DIR) \
			 test_pcap=$(TEST_PCAP) \
			 test_pcap_dir=$(TEST_PCAP_DIR) \
			 test_pps=$(TEST_PPS) \
			 pkt_num=$(PKT_NUM) \
			 super_spines=1 \
			 pods=1 \
			 spines=1 \
			 leafs=1 \
			 hosts_per_leaf=1

MN_TOPO_CLASS := dunefattree
MN_TOPO := --topo=$(MN_TOPO_CLASS),$(call join_with_comma,$(TOPO_ARGS))
MN_SWITCH := --switch=p4simpleswitchgrpc,log_level=$(LOG_LEVEL)
MN_CONTROLLER := --controller=p4controller,topo_class=$(MN_TOPO_CLASS),topo=$(CUSTOM_TOPOLOGY),log_dir=$(LOG_DIR),log_level=$(LOG_LEVEL)
MN_LINK := --link=p4link
MN_TEST := 

MN_LOG_LEVEL := -v $(shell echo $(LOG_LEVEL) | tr '[:upper:]' '[:lower:]')

MN_ARP_FLAG ?= true
MN_ARP := $(if $(filter true,$(MN_ARP_FLAG)),--arp,)

MN_ARGS := $(MN_CUSTOM) \
		   $(MN_TOPO) \
		   $(MN_HOST) \
		   $(MN_SWITCH) \
		   $(MN_CONTROLLER) \
		   $(MN_LINK) \
		   $(MN_LOG_LEVEL) \
		   $(MN_ARP)


PROCESS_PCAPS := ./utils/process_result_pcaps.sh
# Definido antes do bloco condicional pois SEQUENCE_ARG (:=) o expande de imediato.
MODEL_SEQUENCE ?= ../training/model_sequencing/src/results/TON-IOT_model_sequence.json
# CLUSTER_INFO e MODELS_TRAIN_DIR seguem o PARTITIONING_TYPE:
# ofsp/ensemble -> results_ofsp ; default -> results. Os modelos treinados ficam
# no subdiretorio models/ dentro do diretorio de resultados do cluster_analysis.
ifeq ($(filter $(PARTITIONING_TYPE),ofsp ensemble),)
CLUSTER_INFO ?= ../training/cluster_analysis/src/results/perf_results/cluster_info_df.csv
MODELS_TRAIN_DIR ?= ../training/cluster_analysis/src/results/models
# No modo default a ordem dos submodelos vem do model_sequencing.
SEQUENCE_ARG := --sequence-file $(MODEL_SEQUENCE)
else
CLUSTER_INFO ?= ../training/cluster_analysis/src/results_ofsp/perf_results/cluster_info_df.csv
MODELS_TRAIN_DIR ?= ../training/cluster_analysis/src/results_ofsp/models
# No modo ofsp (ensemble) nao ha sequenciamento: todas as views votam.
SEQUENCE_ARG :=
endif

# Comando que (re)gera os programas P4 e o ton.json a partir dos submodelos
# treinados, ja apontando para as pastas do training conforme o modo. O proprio
# generate_p4.py compila os P4 gerados para p4objects ao final.
GENERATE_P4 := python3 $(SOURCES_DIR)/generate_p4.py \
    --cluster-info $(CLUSTER_INFO) \
    --models-dir $(MODELS_TRAIN_DIR) \
    --output-dir $(GENERATED_P4_DIR) \
    --partitioning-type $(PARTITIONING_TYPE) \
    --generate-ton-json $(MODELS) \
    $(SEQUENCE_ARG)
COMPUTE_SCORES := DUNE_RESULTS_DIR=$(RESULTS_DIR) python3 ./utils/calculate_score.py --results $(COMBINED_PCAPS_FILE) --ground-truth $(GROUND_TRUTH_FILE) --outfile $(RESULTS_FILE) --cluster-info $(CLUSTER_INFO) --sequence-file $(MODEL_SEQUENCE) --partitioning-type $(PARTITIONING_TYPE)

# Compute the next available tarball
BASENAME := experiment
NEXT_ARCHIVE_FILE := $(shell \
    i=1; \
    while [ -e "$(CURDIR)/$(BASENAME)_$$i.tar.gz" ]; do \
        i=$$((i+1)); \
    done; \
    echo "$(CURDIR)/$(BASENAME)_$$i.tar.gz" \
)
ARCHIVE_EXPERIMENT := tar -cvzf $(NEXT_ARCHIVE_FILE) \
		                $(PCAP_DIR) \
				$(LOG_DIR) \
				Makefile \
				$(RESULTS_FILE)

DB_SAVE :=  python utils/insert_to_db.py \
	    --file $(RESULTS_FILE) \
	    --archive $(NEXT_ARCHIVE_FILE)


.PHONY: all
all: build


EXPECTED_PCAPS := $(shell awk -F'=' '/^number_of_hosts/ {gsub(/[ \t]+/, "", $$2); print $$2}' utils/params.ini)
ACTUAL_PCAPS := $(words $(wildcard $(TEST_PCAP_DIR)/TON-IOT_*.pcap))

ifneq ($(EXPECTED_PCAPS),$(ACTUAL_PCAPS))
FORCE_PREPARE = FORCE
endif

$(TEST_PCAP_DIR)/.pcaps_prepared: ./utils/prepare_pcap_traces.py utils/params.ini $(TEST_PCAP) $(GROUND_TRUTH_FILE) $(FORCE_PREPARE)
	@if [ "$(EXPECTED_PCAPS)" -eq "$(ACTUAL_PCAPS)" ] && [ -n "$$(ls $(TEST_PCAP_DIR)/TON-IOT_*.pcap 2>/dev/null)" ] && [ ! utils/params.ini -nt $$(ls -t $(TEST_PCAP_DIR)/TON-IOT_*.pcap | head -1) ]; then \
		echo "Pcaps are already generated and up to date. Skipping generation."; \
	else \
		echo "Generating $(EXPECTED_PCAPS) pcap traces..."; \
		python3 ./utils/prepare_pcap_traces.py; \
	fi
	@ACTUAL=$$(ls -1 $(TEST_PCAP_DIR)/TON-IOT_*.pcap 2>/dev/null | wc -l); \
	if [ "$$ACTUAL" -ne "$(EXPECTED_PCAPS)" ]; then \
		echo "Error: Expected $(EXPECTED_PCAPS) pcap files, but found $$ACTUAL in $(TEST_PCAP_DIR)."; \
		exit 1; \
	fi
	@touch $@

FORCE:

$(TEST_PCAP) $(GROUND_TRUTH_FILE):
	@echo "Error: Required file $@ not found. Please provide it or update the configuration."
	@exit 1

.PHONY: run
run: build $(TEST_PCAP) $(GROUND_TRUTH_FILE) | $(RUN_DIRS)
	bash -o pipefail -c '$(MN) $(MN_ARGS) 2>&1 | tee "$(LOG_DIR)/mn.log"; ec=$$?; if grep -qE "Traceback|Caught exception" "$(LOG_DIR)/mn.log"; then exit 1; else exit $$ec; fi'

.PHONY: results
results:
	   $(MKDIR) $(RESULTS_DIR)
	   $(PROCESS_PCAPS)
	   $(COMPUTE_SCORES)
	   cp $(COMBINED_PCAPS_FILE) $(RESULTS_DIR)/
	   -capinfos -M $(TEST_PCAP) > $(RESULTS_DIR)/capinfos_input.txt 2>/dev/null
	   -capinfos -M $(PCAP_DIR)/pe_l*-eth*_out.pcap > $(RESULTS_DIR)/capinfos_output.txt 2>/dev/null
	   # tcpreplay stats intencionalmente nao salvos: drop medido por capinfos (input vs saida do ultimo switch).
	   # -cp $(LOG_DIR)/tcpreplay_stats.txt $(RESULTS_DIR)/ 2>/dev/null
	   ln -sfn run_$(RUN_ID) $(RESULTS_BASE)/latest
	   @echo "Results saved to $(RESULTS_DIR)"
# 	   $(ARCHIVE_EXPERIMENT)
ifeq ($(SAVE_RESULTS),YES)
	   $(DB_SAVE)
	   @echo Saving results to DB
else
	   @echo Skipping saving results to DB
endif

.PHONY: run-test
run-test: override MN_ARGS += --test=tonfattree
run-test: $(TEST_PCAP_DIR)/.pcaps_prepared run
	$(MAKE) results

.PHONY: run-smoke-test
run-smoke-test: override PKT_NUM := 1000
run-smoke-test: override MN_ARGS := $(MN_CUSTOM) --topo=$(MN_TOPO_CLASS),$(call join_with_comma,$(TOPO_ARGS_LINEAR)) $(MN_HOST) $(MN_SWITCH) $(MN_CONTROLLER) $(MN_LINK) $(MN_LOG_LEVEL) --test=tonlinear
run-smoke-test: run
	$(MAKE) results

TOPO_ARGS_SIMPLE = models=configs/models/ton_1st_model_only.json \
			 models_dir=$(MODELS_DIR) \
			 objects_dir=$(OBJECTS_DIR) \
			 log_dir=$(LOG_DIR) \
			 pcap_dir=$(PCAP_DIR) \
			 test_pcap=$(TEST_PCAP) \
			 test_pcap_dir=$(TEST_PCAP_DIR) \
			 test_pps=$(TEST_PPS) \
			 pkt_num=$(PKT_NUM)

.PHONY: run-simple-test
run-simple-test: override PKT_NUM := 50000
run-simple-test: override MN_ARGS := $(MN_CUSTOM) --topo=dunesimple,$(call join_with_comma,$(TOPO_ARGS_SIMPLE)) $(MN_HOST) $(MN_SWITCH) --controller=p4controller,topo_class=dunesimple,topo=$(CUSTOM_TOPOLOGY),log_dir=$(LOG_DIR),log_level=$(LOG_LEVEL) $(MN_LINK) $(MN_LOG_LEVEL) --test=tonsimple
run-simple-test: run

.PHONY: run-ton-iot-test
# run-ton-iot-test: override PKT_NUM := 1000000
run-ton-iot-test: override MN_ARGS := $(MN_CUSTOM) --topo=$(MN_TOPO_CLASS),$(call join_with_comma,$(TOPO_ARGS_LINEAR)) $(MN_HOST) $(MN_SWITCH) $(MN_CONTROLLER) $(MN_LINK) $(MN_LOG_LEVEL) --test=tonlinear
run-ton-iot-test: run
	$(MAKE) results


.PHONY: run-linear-test
run-linear-test: override MN_ARGS := $(MN_CUSTOM) --topo=$(MN_TOPO_CLASS),$(call join_with_comma,$(TOPO_ARGS_LINEAR)) $(MN_HOST) $(MN_SWITCH) $(MN_CONTROLLER) $(MN_LINK) $(MN_LOG_LEVEL) --test=tonlinear
run-linear-test: run
	$(MAKE) results

.PHONY: run-table-5
run-table-5:
	$(MAKE) run-test PODS=3 SPINES=4 LEAFS=4 SUPER_SPINES=1 HOSTS_PER_LEAF=2 TEST_PPS=100

.PHONY: stop
stop:
	$(MN) -c

.PHONY: build
build: generate
	$(MAKE) --no-print-directory compile

# Gera os programas P4 e o ton.json a partir dos submodelos treinados.
.PHONY: generate
generate:
	$(GENERATE_P4)

# Compila os P4 (gerados + built_in) para BMv2 JSON. Alvo separado para que o
# wildcard de SOURCES seja reavaliado apos a geracao, via sub-make.
.PHONY: compile
compile: $(OBJECTS)


$(OBJECTS_DIR)/%.json: %.p4 | $(OBJECTS_DIR)
	$(P4C) $(P4_FLAGS) $(P4_FLAGS_RUNTIME) $(P4_FLAGS_DEPS) -o $@ $<


$(OBJECTS_DIR):
	$(MKDIR) $(OBJECTS_DIR)


$(RUN_DIRS):
	$(MKDIR) $(RUN_DIRS)



.PHONY: clean
clean: stop
	$(RMDIR) $(RUN_DIRS)
	$(RM) $(FATTREE_TOPOLOGY)

.PHONY: aggregate
aggregate:
	python3 ./utils/aggregate_results.py --type $(PARTITIONING_TYPE) --results-base ./results

.PHONY: clean-all
clean-all: clean
	$(RMDIR) $(OBJECTS_DIR)
	$(RMDIR) ./results


-include $(DEPS)

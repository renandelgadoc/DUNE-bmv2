# DUNE-bmv2: Distributed In-Network Inference with Feature-Set Partitioning

> ⚠️ **Disclaimer & Attribution:** This repository builds on top of the original **DUNE** framework by the NDS group (IMDEA Networks Institute). Before anything else, please read the **[NOTICE](NOTICE)** file for licensing, attribution, and dataset-usage terms.

This repository contains the source code developed for an undergraduate thesis (TCC) that extends **DUNE**, a framework for distributed Machine Learning (ML) inference in the user plane. It includes both the ML training pipeline and a software-switch (BMv2) implementation that reproduces the data-plane execution in Mininet.

## Overview

Running ML models directly in the user plane enables line-rate in-network inference, cutting latency and improving scalability for use cases like traffic monitoring and intrusion detection. However, fitting ML models into programmable network devices is hard because of their strict memory and computing constraints.

Instead of forcing a monolithic model into a single device, DUNE distributes the inference across **multiple programmable switches**. It automatically (i) breaks a large ML model into simpler sub-models that preserve accuracy while minimizing resource usage, and (ii) designs and sequences these sub-models to enable an efficient distributed execution of joint packet- and flow-level inference.

The original DUNE partitions the problem **by class** (SPP — *Set Partitioning Problem*): each sub-model handles a disjoint subset of classes and forwards uncertain packets down a rigid sequential chain, where an early mistake becomes a definitive decision.

This work adds an alternative partitioning strategy, **OFSP** (*Optimal Feature Set Partitioning*): the dataset is split **by features** so that every sub-model can classify *all* classes from a partial view of the traffic. The switches then aggregate their independent predictions through a **majority-voting ensemble**, removing the rigid sequential dependency. Both strategies are trained by the same pipeline and can be compared head-to-head on the ToN_IoT dataset.

## Repository Structure

The project is organized as a linear pipeline: each ML step lives in its own directory under `training/` and depends on the output of the previous one, while `simulation/` runs the resulting models on the data plane.

### `training/` — ML pipeline
1. **`data_generation/`** — Prepares train/test data from the source PCAPs (ground-truth CSVs and flow packet counts).
2. **`unconstrained_model_analysis/`** — Trains a single unconstrained model and extracts per-class feature importance.
3. **`model_partitioning/`** — SPP solver that splits the task into class-based clusters (baseline DUNE strategy).
4. **`ofsp_partitioning/`** — Feature-based partitioning solver (OFSP), the contribution of this work.
5. **`cluster_analysis/`** — Trains the per-cluster Random Forest sub-models and evaluates their F1 score.
6. **`model_sequencing/`** — Orders the sub-models across switches to optimize inference performance.
7. **`dune_tutorial.ipynb`** — Walkthrough notebook covering the pipeline steps.

*Each module has its own `src/params.ini` for configuration.*

### `simulation/` — BMv2 data plane
- **`p4sources/`** — P4 data-plane programs and the Jinja2 generator (`generate_p4.py`), including the majority-voting inference used by OFSP.
- **`dune/`** — Custom Mininet topology and P4 node/link implementations.
- **`configs/`** — Topology and model configuration (JSON).
- **`controller.py`** — Main switch controller (gRPC + P4Runtime).
- **`utils/`** — Orchestration, scoring, and PCAP-preparation helpers.
- **`Makefile`** — Build and test targets.

## Dependencies & Installation

This project relies on the P4 development environment. We use the scripts from the [p4-guide](https://github.com/jafingerhut/p4-guide) repository to install all dependencies (PI, BMv2, P4C, Mininet):

```bash
git clone https://github.com/jafingerhut/p4-guide.git
./p4-guide/bin/install-p4dev-v10.sh
```

**⚠️ Known issue:** The `behavioral-model` Thrift interface has a bug ([p4lang/behavioral-model#1367](https://github.com/p4lang/behavioral-model/pull/1367)) that prevents the controller from connecting to the switches. Until it is fixed upstream, the corresponding patch must be applied.

Activate the Python virtual environment created by the installation script before running any command or make target:

```bash
source ~/p4-guide/p4dev-python-venv/bin/activate
```

The ML pipeline steps under `training/` isolate their dependencies per directory:

```bash
pip install -r <directory_name>/requirements.txt
```

## Required Data Files

The ToN_IoT dataset files are **not** distributed with this repository. Obtain them from the [original source](https://research.unsw.edu.au/projects/toniot-datasets) (free for academic research with citation) and place them in a `data/` directory, following the paths expected in each step's `params.ini` and in `simulation/utils/params.ini`.

## Quick Start

From `simulation/`, build the P4 sources and run a minimal smoke test to check the environment:

```bash
make build
make run-smoke-test
```

Other useful targets:

- `make run-ton-iot-test` — run the ToN_IoT intrusion-detection experiment.
- `make run-table-5` — reproduce the DUNE paper results (Table V).
- `make run-linear-test` / `make run-test` — linear and fat-tree topology tests.

## Citation

This work extends DUNE. If you use this code or framework in your research, please cite the original INFOCOM 2025 paper:

```bibtex
@INPROCEEDINGS{11044678,
  author={Bütün, Beyza and De Andres Hernandez, David and Gucciardo, Michele and Fiore, Marco},
  booktitle={IEEE INFOCOM 2025 - IEEE Conference on Computer Communications},
  title={DUNE: Distributed Inference in the User Plane},
  year={2025},
  volume={},
  number={},
  pages={1-10},
  keywords={Sequential analysis;Accuracy;Computational modeling;Scalability;Memory management;Machine learning;Hardware;Delays;Resource management;Monitoring},
  doi={10.1109/INFOCOM55648.2025.11044678}
}
```

#!/usr/bin/env python3
"""Generate P4 source files from trained RF models and cluster info."""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import yaml
from ast import literal_eval
from jinja2 import Environment, FileSystemLoader
import subprocess

# Add project root to path so we can import convert_RF_and_populate_tables
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_feature_map(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


# Legacy inline map kept for reference; the script now loads from feature_map.yml
FEATURE_MAP = {
    "ip.len": {
        "type": "packet",
        "struct_field": "ip_len",
        "bit_width": 16,
        "key_expr": "features.ip_len",
        "extract": "features.ip_len = hdr.ipv4.total_length;",
    },
    "ip.ttl": {
        "type": "packet",
        "struct_field": "ip_ttl",
        "bit_width": 8,
        "key_expr": "features.ip_ttl",
        "extract": "features.ip_ttl = hdr.ipv4.ttl;",
    },
    "srcport": {
        "type": "packet",
        "struct_field": "src_port",
        "bit_width": 16,
        "key_expr": "features.src_port",
        "extract": "features.src_port = meta.src_port;",
    },
    "dstport": {
        "type": "packet",
        "struct_field": "dst_port",
        "bit_width": 16,
        "key_expr": "features.dst_port",
        "extract": "features.dst_port = meta.dst_port;",
    },
    "tcp.window_size_value": {
        "type": "packet",
        "struct_field": "tcp_window",
        "bit_width": 16,
        "key_expr": "features.tcp_window",
        "extract": (
            "if (hdr.ipv4.protocol == IPv4Proto.TCP) {\n"
            "            features.tcp_window = hdr.tcp.window;\n"
            "        } else {\n"
            "            features.tcp_window = 0;\n"
            "        }"
        ),
    },
    "tcp.hdr_len": {
        "type": "packet",
        "struct_field": "tcp_len",
        "bit_width": 4,
        "key_expr": "features.tcp_len",
        "extract": (
            "if (hdr.ipv4.protocol == IPv4Proto.TCP) {\n"
            "            features.tcp_len = hdr.tcp.data_offset;\n"
            "        } else {\n"
            "            features.tcp_len = 0;\n"
            "        }"
        ),
    },
    "tcp.flags.ack": {
        "type": "packet",
        "struct_field": "tcp_ack",
        "bit_width": 1,
        "key_expr": "features.tcp_ack",
        "extract": (
            "if (hdr.ipv4.protocol == IPv4Proto.TCP) {\n"
            "            features.tcp_ack = hdr.tcp.ack;\n"
            "        } else {\n"
            "            features.tcp_ack = 0;\n"
            "        }"
        ),
    },
    "udp.length": {
        "type": "packet",
        "struct_field": "udp_len",
        "bit_width": 16,
        "key_expr": "features.udp_len",
        "extract": (
            "if (hdr.ipv4.protocol == IPv4Proto.UDP) {\n"
            "            features.udp_len = hdr.udp.length;\n"
            "        } else {\n"
            "            features.udp_len = 0;\n"
            "        }"
        ),
    },
    "Packet Length Total": {
        "type": "stateful",
        "struct_field": "total_length",
        "bit_width": 16,
        "key_expr": "statefull_features.total_length",
        "reg_name": "flows_total_length",
        "update_logic": "total_length",
    },
    "Min Packet Length": {
        "type": "stateful",
        "struct_field": "min_length",
        "bit_width": 16,
        "key_expr": "statefull_features.min_length",
        "reg_name": "flows_min_length",
        "update_logic": "min_length",
    },
    "Max Packet Length": {
        "type": "stateful",
        "struct_field": "max_length",
        "bit_width": 16,
        "key_expr": "statefull_features.max_length",
        "reg_name": "flows_max_length",
        "update_logic": "max_length",
    },
    "FIN Flag Count": {
        "type": "stateful",
        "struct_field": "fin_flag_count",
        "bit_width": 8,
        "key_expr": "statefull_features.fin_flag_count",
        "reg_name": "flows_fin_flag_count",
        "update_logic": "fin_flag_count",
    },
    "PSH Flag Count": {
        "type": "stateful",
        "struct_field": "psh_flag_count",
        "bit_width": 8,
        "key_expr": "statefull_features.psh_flag_count",
        "reg_name": "flows_psh_flag_count",
        "update_logic": "psh_flag_count",
    },
    "ACK Flag Count": {
        "type": "stateful",
        "struct_field": "ack_flag_count",
        "bit_width": 8,
        "key_expr": "statefull_features.ack_flag_count",
        "reg_name": "flows_ack_flag_count",
        "update_logic": "ack_flag_count",
    },
}


def get_tree_code_sizes(clf, feature_names):
    """Get the codeword bit width for each feature in each tree.
    Returns tree_code_sizes[tree_id][feature_idx] = bit width."""
    from generate_tables_entries_txt import get_splits, get_feature_table, get_feature_codes_with_ranges

    splits = get_splits(clf, feature_names)
    n_trees = len(clf.estimators_)
    tree_code_sizes = [[] for _ in range(n_trees)]
    for fea_name in feature_names:
        feature_table = get_feature_table(splits, fea_name)
        Ranges, Codes = get_feature_codes_with_ranges(feature_table, n_trees)
        for j in range(n_trees):
            tree_code_sizes[j].append(len(Codes.iloc[0, j]) - 2)
    return tree_code_sizes


def get_feature_table_sizes(clf, feature_names):
    """Get the number of entries needed for each feature table."""
    from generate_tables_entries_txt import get_splits, get_feature_table

    splits = get_splits(clf, feature_names)
    sizes = []
    for fea_name in feature_names:
        feature_table = get_feature_table(splits, fea_name)
        sizes.append(len(feature_table))
    return sizes


def build_model_context(model_path, cluster_id, cluster_row, class_offset, inference_point, feature_map):
    """Build the Jinja2 template context for a single model."""
    clf = pd.read_pickle(model_path)
    feature_names = clf.feature_names_in_.tolist()
    n_trees = len(clf.estimators_)

    tree_code_sizes = get_tree_code_sizes(clf, feature_names)
    table_sizes = get_feature_table_sizes(clf, feature_names)

    # Separate features into packet-level and stateful
    packet_features = []
    stateful_features = []
    features_info = []

    for i, fname in enumerate(feature_names):
        fmap = feature_map[fname]
        info = {
            "index": i,
            "name": fname,
            "code_sizes_per_tree": [tree_code_sizes[t][i] for t in range(n_trees)],
            # Trees that actually split on this feature (non-zero codeword). The
            # SetCode action only declares/assigns params for these; a tree with a
            # 0-bit code is omitted so P4Runtime never has to write a bit<0> param.
            "used_codes": [t for t in range(n_trees) if tree_code_sizes[t][i] > 0],
            "table_size": table_sizes[i],
            "key_expr": fmap["key_expr"],
        }
        features_info.append(info)

        if fmap["type"] == "packet":
            packet_features.append(fmap)
        else:
            stateful_features.append(fmap)

    # Number of real classes (excluding Other)
    classes = literal_eval(cluster_row["Class List"]) if isinstance(cluster_row["Class List"], str) else cluster_row["Class List"]
    n_classes = len(classes) + 1  # +1 for Other
    unknown_class_id = class_offset + n_classes

    # N_Leaves for code table size
    n_leaves_raw = cluster_row["N_Leaves"]
    if isinstance(n_leaves_raw, str):
        n_leaves_raw = literal_eval(n_leaves_raw)
    if isinstance(n_leaves_raw, list):
        n_leaves = max(n_leaves_raw)
    else:
        n_leaves = int(n_leaves_raw)

    # # N_Leaves for code table size — count from actual model, not stale CSV
    # from generate_tables_entries_txt import get_codes_and_masks as _gcam
    # n_leaves = max(
    #     len(_gcam(tree, feature_names)[0])
    #     for tree in clf.estimators_
    # )

    context = {
        "cluster_id": cluster_id,
        "model_id": cluster_id + 1,  # placeholder; overridden with position+1 in main loop
        "inference_point": inference_point,
        "n_trees": n_trees,
        "n_leaves": n_leaves,
        "feature_names": feature_names,
        "features_info": features_info,
        "packet_features": packet_features,
        "stateful_features": stateful_features,
        "tree_code_sizes": tree_code_sizes,
        "table_sizes": table_sizes,
        "class_offset": class_offset,
        "unknown_class_id": unknown_class_id,
        "n_classes": n_classes,
    }
    return context, n_classes


def generate_ton_json(cluster_info, sequence, models_subdir="ton"):
    """Generate configs/models/ton.json from cluster info and model sequence."""
    import json
    ton = {}
    prev = None
    for position, cluster_id in enumerate(sequence):
        key = f"m{position}"
        ton[key] = {
            "p4": f"ton_iot_m{cluster_id}",
            "files": [f"{models_subdir}/selected_model_cluster{cluster_id}.pkl"],
            "previous": prev,
        }
        prev = key
    return ton


def main():
    parser = argparse.ArgumentParser(description="Generate P4 from trained RF models")
    parser.add_argument("--cluster-info", required=True, help="Path to cluster_info_df.csv")
    parser.add_argument("--models-dir", required=True, help="Directory containing selected_model_cluster*.pkl")
    parser.add_argument("--output-dir", default=".", help="Output directory for generated P4 files")
    parser.add_argument("--inference-point", type=int, default=3, help="Inference point (npkts)")
    parser.add_argument("--use-case", default="TON-IOT", help="Use case name for output files")
    parser.add_argument("--sequence-file", default=None,
                        help="Path to model_sequence.json from model_sequencing (optional, defaults to cluster order)")
    parser.add_argument("--generate-ton-json", default=None,
                        help="Output path for ton.json (e.g., configs/models/ton.json)")
    parser.add_argument("--partitioning-type", default="default", choices=["default", "ensemble", "ofsp"],
                        help="Partitioning type: 'default' (class subsets per switch) or 'ensemble'/'ofsp' (all classes per switch, majority vote)")
    args = parser.parse_args()

    # 'ofsp' e 'ensemble' sao o mesmo modo; mantemos o nome de diretorio 'ofsp'.
    dir_name = "default" if args.partitioning_type == "default" else "ofsp"
    if args.partitioning_type == "ofsp":
        args.partitioning_type = "ensemble"
    # Deriva o output-dir do modo quando nao foi passado explicitamente.
    if args.output_dir == ".":
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"p4_{dir_name}")
        print(f"Output dir derivado do modo ({dir_name}): {args.output_dir}")

    # Load cluster info
    cluster_info = pd.read_csv(args.cluster_info)

    # Determine cluster processing order
    if args.sequence_file:
        import json
        with open(args.sequence_file, 'r') as f:
            sequence = json.load(f)
        print(f"Using model sequence from {args.sequence_file}: {sequence}")
    else:
        sequence = cluster_info["Cluster"].tolist() if "Cluster" in cluster_info.columns else list(range(len(cluster_info)))
        print(f"Using default cluster order: {sequence}")

    # Generate ton.json if requested
    if args.generate_ton_json:
        import json
        ton = generate_ton_json(cluster_info, sequence)
        os.makedirs(os.path.dirname(args.generate_ton_json), exist_ok=True)
        with open(args.generate_ton_json, 'w') as f:
            json.dump(ton, f, indent=4)
            f.write('\n')
        print(f"Generated {args.generate_ton_json}")

    # Load feature map from YAML
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    feature_map = load_feature_map(os.path.join(templates_dir, "feature_map.yml"))

    # Set up Jinja2
    env = Environment(loader=FileSystemLoader(templates_dir), trim_blocks=True, lstrip_blocks=True)
    template = env.get_template("model.p4.j2")

    n_models = len(sequence)
    partitioning_type = args.partitioning_type

    # For ensemble mode, get the number of classes from any model. Every model
    # in the ensemble is trained on the full class set and votes over it, so the
    # vote tally (Votes_t) and the dispatch/majority logic are sized by n_classes.
    ensemble_n_classes = None
    if partitioning_type == "ensemble":
        first_cluster = int(sequence[0])
        first_model_path = os.path.join(args.models_dir, f"selected_model_cluster{first_cluster}.pkl")
        if os.path.exists(first_model_path):
            first_clf = pd.read_pickle(first_model_path)
            ensemble_n_classes = len(first_clf.classes_)
        print(f"Ensemble mode: {n_models} models, {ensemble_n_classes} classes")

    # For ensemble mode, generate shared include files (headers, inference, ingress)
    if partitioning_type == "ensemble":
        shared_context = {
            "partitioning_type": partitioning_type,
            "n_models": n_models,
            "n_classes": ensemble_n_classes,
        }
        for tmpl_name in ["dune_headers.p4.j2", "dune_inference.p4.j2", "dune_ingress.p4.j2"]:
            tmpl = env.get_template(tmpl_name)
            out_name = tmpl_name.replace(".j2", "")
            rendered = tmpl.render(**shared_context)
            out_path = os.path.join(args.output_dir, out_name)
            with open(out_path, "w") as f:
                f.write(rendered)
            print(f"Generated {out_path}")

    # Generate P4 files in sequence order (class offsets follow this order)
    class_offset = 0
    for position, cluster_id in enumerate(sequence):
        cluster_id = int(cluster_id)
        row = cluster_info[cluster_info["Cluster"] == cluster_id].iloc[0] if "Cluster" in cluster_info.columns else cluster_info.iloc[cluster_id]
        model_path = os.path.join(args.models_dir, f"selected_model_cluster{cluster_id}.pkl")

        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found, skipping cluster {cluster_id}")
            continue

        context, n_classes = build_model_context(
            model_path, cluster_id, row, class_offset, args.inference_point, feature_map
        )

        context["partitioning_type"] = partitioning_type
        context["n_models"] = n_models
        context["model_position"] = position
        # MODEL_ID gates the dependency chain (a model runs only when the packet's
        # model_id == MODEL_ID - 1). Packets start at model_id 0 and traverse the
        # switches in sequence/position order, so MODEL_ID must follow the position
        # in the sequence, NOT the cluster_id. Using cluster_id (the previous
        # behaviour, see build_model_context) desynchronised the chain whenever the
        # sequence wasn't [0,1,2,..], making most models never execute.
        context["model_id"] = position + 1
        context["is_last_model"] = (position == n_models - 1)
        if partitioning_type == "ensemble":
            context["n_classes"] = ensemble_n_classes

        output = template.render(**context)
        output_file = os.path.join(args.output_dir, f"ton_iot_m{cluster_id}.p4")
        with open(output_file, "w") as f:
            f.write(output)

        print(f"Generated {output_file} (cluster {cluster_id}, {len(context['feature_names'])} features, "
              f"{context['n_trees']} trees, {context['n_leaves']} leaves, "
              f"{'ensemble pos=' + str(position) if partitioning_type == 'ensemble' else 'class_offset=' + str(class_offset)})")

        class_offset += n_classes - 1  # Don't count Other in offset

    # Compile generated P4 files to BMv2 JSON
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_root = os.path.dirname(script_dir)
    include_dir = os.path.join(sim_root, "p4include")
    objects_dir = os.path.join(sim_root, "p4objects")
    os.makedirs(objects_dir, exist_ok=True)
    for p4_file in sorted(os.listdir(args.output_dir)):
        if not p4_file.endswith(".p4"):
            continue
        # dune_*.p4 are shared includes (headers/inference/ingress), not
        # standalone programs, so they must not be compiled on their own.
        if p4_file.startswith("dune_"):
            continue
        p4_path = os.path.join(args.output_dir, p4_file)
        base = p4_file[:-len(".p4")]
        json_out = os.path.join(objects_dir, base + ".json")
        p4info_out = os.path.join(objects_dir, base + ".p4.p4info.txtpb")
        dep_file = os.path.join(objects_dir, base + ".d")
        cmd = [
            "p4c-bm2-ss", "--std", "p4-16",
            "-I", include_dir,
            "--p4runtime-files", p4info_out,
            "-MD", "-MP", "-MT", json_out, "-MF", dep_file,
            "-o", json_out,
            p4_path,
        ]
        print(f"Compiling {p4_file} ...")
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"ERROR: compilation failed for {p4_file}")
            sys.exit(result.returncode)
    print("Build complete.")


if __name__ == "__main__":
    main()

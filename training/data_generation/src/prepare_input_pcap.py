import os
import sys
import glob
import signal
import shutil
import subprocess
from multiprocessing import Pool, cpu_count
from scapy.all import PcapReader, PcapWriter, Raw, IP
from scapy.layers.l2 import CookedLinux, Ether

SRC_MAC = "00:00:00:00:00:01"
DST_MAC = "00:00:00:00:00:02"
BATCH_SIZE = "100000"
MAX_CORES = max(1, cpu_count() - 1)
TEMP_DIR = "temp_pcaps"
PROCESSED_DIR = "processed_pcaps"

INPUT_DIR = "pcaps_not_processed"
TRAIN_DIR = "train_pcaps"
TEST_DIR = "test_pcaps"
TRAIN_RATIO = 0.75

KNOWN_CLASSES = [
    "ddos", "dos", "injection", "mitm", "xss",
    "scanning", "password", "runsomware", "ransomware", "backdoor",
]

# Map each input directory to its corresponding output directory
INPUT_OUTPUT_DIRS = [
    ("test_pcaps_not_processed", "test_pcaps"),
    ("train_pcaps_not_processed", "train_pcaps"),
]


def extract_class_from_filename(filename):
    name_lower = filename.lower().replace(".pcap", "").replace(".pcapng", "")
    for cls in KNOWN_CLASSES:
        if cls in name_lower:
            return cls
    # "normal IoT" captures (normal_IoT_*.pcap) are split separately from the
    # regular "normal" captures so both are represented in train and test.
    if "iot" in name_lower:
        return "normal_iot"
    return "normal"

def cleanup_and_exit(signum=None, frame=None):
    """
    Signal handler to ensure the temporary directory is wiped 
    if the script is interrupted or terminated.
    """
    if signum:
        print(f"\n\n[!] Received signal {signum}. Halting execution...")
    else:
        print("\n\n[!] An error occurred. Halting execution...")
        
    print(f"Cleaning up temporary directory: {TEMP_DIR}/")
    if os.path.exists(TEMP_DIR):
        try:
            # rmtree forces deletion of the directory and all files inside
            shutil.rmtree(TEMP_DIR, ignore_errors=True)
            print("Cleanup successful.")
        except Exception as e:
            print(f"Failed to remove {TEMP_DIR}: {e}")
            
    sys.exit(1 if signum else 0)

def process_file(in_file):
    """
    Worker function: Reads a chunk, modifies it, 
    writes to a new file, and deletes the input chunk.
    """
    out_file = in_file.replace("split_", "processed_")
    
    with PcapReader(in_file) as pcap_in, PcapWriter(out_file, append=False, sync=True) as pcap_out:
        for pkt in pcap_in:
            try:
                if pkt.haslayer(CookedLinux):
                    sll = pkt[CookedLinux]
                    inner = sll.payload
                else:
                    inner = pkt

                # Check for an actual IP layer by content instead of
                # trusting sll.proto == 0x0800, which misses packets
                # where IP/TCP sits right after the link layer but
                # the proto field doesn't say so (and incidentally
                # also drops loopback frames, since those don't carry
                # a real IP layer either).
                if inner.haslayer(IP):
                    # Drop ICMP (IP proto 1): it has no L4 ports, so it cannot
                    # form a flow (the Flow ID is built from the 5-tuple incl.
                    # ports) and is unusable downstream. Remove it up front.
                    if inner[IP].proto == 1:
                        continue
                    new_pkt = Ether(src=SRC_MAC, dst=DST_MAC, type=0x0800) / Raw(bytes(inner[IP]))
                    if hasattr(pkt, "time"):
                        new_pkt.time = pkt.time
                    pcap_out.write(new_pkt)
            except Exception:
                continue

    return out_file

def run_pcapfix(infile, fixed_file):
    """
    Runs pcapfix on the input file to repair any structural
    issues before processing. Returns the path to the fixed file,
    or the original file if pcapfix found nothing to fix (no output created).
    """
    cmd = ["pcapfix", "-o", fixed_file, infile]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    if os.path.exists(fixed_file):
        return fixed_file
    else:
        print(f"  pcapfix found nothing to fix, using original file.")
        return infile

def process_pcap(infile, outfile):
    """
    Full pipeline for a single pcap file:
    pcapfix → editcap split → parallel SLL-to-Ether conversion → mergecap.
    """
    basename = os.path.basename(infile)
    print(f"\n{'='*60}")
    print(f"Processing: {basename}")
    print(f"  Input:  {infile}")
    print(f"  Output: {outfile}")
    print(f"{'='*60}")

    os.makedirs(TEMP_DIR, exist_ok=True)

    try:
        # STEP 0: PCAPFIX
        print("\n  STEP 0: RUNNING PCAPFIX")
        print("  " + "#" * 26)
        fixed_file = os.path.join(TEMP_DIR, f"fixed_{basename}")
        fixed_file = run_pcapfix(infile, fixed_file)
        print(f"  pcapfix complete: {fixed_file}\n")

        # STEP 1: SPLITTING WITH EDITCAP
        print("  STEP 1: SPLITTING WITH EDITCAP")
        print("  " + "#" * 30)
        split_base = os.path.join(TEMP_DIR, "split.pcap")

        cmd_split = ["editcap", "-c", BATCH_SIZE, fixed_file, split_base]
        print(f"  Running: {' '.join(cmd_split)}")
        subprocess.run(cmd_split, check=True)

        batch_files = sorted(glob.glob(os.path.join(TEMP_DIR, "split*.pcap")))
        total_batches = len(batch_files)
        print(f"  Successfully split into {total_batches} files.\n")

        # STEP 2: PROCESSING BATCHES
        print(f"  STEP 2: PROCESSING BATCHES ({MAX_CORES} cores)")
        print("  " + "#" * 40)
        processed_files = []

        with Pool(processes=MAX_CORES) as pool:
            for i, result_file in enumerate(pool.imap(process_file, batch_files)):
                if os.path.exists(result_file):
                    processed_files.append(result_file)
                percent = ((i + 1) / total_batches) * 100
                sys.stdout.write(f"\r  Progress: {percent:.2f}% completed ({i+1}/{total_batches} batches)")
                sys.stdout.flush()

        if not processed_files:
            print(f"\n  [WARN] No valid packets found in {infile}, skipping merge.")
            return

        # STEP 3: MERGING WITH MERGECAP
        print(f"\n\n  STEP 3: MERGING WITH MERGECAP")
        print("  " + "#" * 29)
        os.makedirs(os.path.dirname(outfile) if os.path.dirname(outfile) else '.', exist_ok=True)

        cmd_merge = ["mergecap", "-w", outfile] + processed_files
        print(f"  Merging {len(processed_files)} files into {outfile}...")
        subprocess.run(cmd_merge, check=True)

        print(f"  Done! Saved to: {outfile}")

    finally:
        # Always clean up temp directory between files
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR, ignore_errors=True)

def get_packet_count(pcap_path):
    """Get packet count from a pcap file using capinfos."""
    result = subprocess.run(
        ["capinfos", "-c", "-M", pcap_path],
        capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if "Number of packets" in line:
            return int(line.split()[-1])
    return 0


def split_processed_pcaps(processed_dir, train_dir, test_dir, train_ratio=TRAIN_RATIO):
    """
    Group processed PCAPs by class (from filename), then split files
    per class into train/test so that ~80% of packets go to train.
    """
    from collections import defaultdict

    pcap_files = sorted(glob.glob(os.path.join(processed_dir, "*.pcap")))
    if not pcap_files:
        print(f"[!] No processed .pcap files found in {processed_dir}")
        return

    # Count packets per file
    print(f"\n  Counting packets in {len(pcap_files)} files...")
    file_counts = {}
    for f in pcap_files:
        count = get_packet_count(f)
        file_counts[f] = count
        print(f"    {os.path.basename(f)}: {count} packets")

    # Group by class
    by_class = defaultdict(list)
    for f in pcap_files:
        cls = extract_class_from_filename(os.path.basename(f))
        by_class[cls].append(f)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Splitting by packet count into train/test ({train_ratio:.0%}/{1-train_ratio:.0%})")
    print(f"{'='*60}")

    for cls in sorted(by_class.keys()):
        files = sorted(by_class[cls], key=lambda f: file_counts[f], reverse=True)
        total_pkts = sum(file_counts[f] for f in files)
        target_train_pkts = total_pkts * train_ratio

        # Accumulate files into train until we reach the target (biggest first)
        train_files = []
        test_files = []
        train_pkts = 0
        for f in files:
            if train_pkts < target_train_pkts and (not train_files or len(files) - len(train_files) > 1):
                train_files.append(f)
                train_pkts += file_counts[f]
            else:
                test_files.append(f)

        # Ensure at least 1 file in each set when possible
        if not test_files and len(train_files) > 1:
            moved = train_files.pop()
            train_pkts -= file_counts[moved]
            test_files.append(moved)

        test_pkts = sum(file_counts[f] for f in test_files)
        actual_ratio = train_pkts / total_pkts if total_pkts > 0 else 0

        print(f"  {cls:>12s}: {len(train_files)} train ({train_pkts} pkts), "
              f"{len(test_files)} test ({test_pkts} pkts) "
              f"[{actual_ratio:.1%}/{1-actual_ratio:.1%}]")

        for f in train_files:
            shutil.copy2(f, os.path.join(train_dir, os.path.basename(f)))
        for f in test_files:
            shutil.copy2(f, os.path.join(test_dir, os.path.basename(f)))

    print(f"\n  Train dir: {train_dir}/")
    print(f"  Test dir:  {test_dir}/")


def main():
    # Register the signal handlers for graceful cleanup
    signal.signal(signal.SIGINT, cleanup_and_exit)   # Catches Ctrl+C
    signal.signal(signal.SIGTERM, cleanup_and_exit)   # Catches standard kill commands

    try:
        for input_dir, output_dir in INPUT_OUTPUT_DIRS:
            if not os.path.isdir(input_dir):
                print(f"[!] Input directory not found, skipping: {input_dir}")
                continue

            os.makedirs(output_dir, exist_ok=True)

            pcap_files = sorted(glob.glob(os.path.join(input_dir, "*.pcap")))
            if not pcap_files:
                print(f"[!] No .pcap files found in {input_dir}, skipping.")
                continue

            print(f"\n{'*'*60}")
            print(f"  Directory: {input_dir}  →  {output_dir}")
            print(f"  Found {len(pcap_files)} pcap file(s)")
            print(f"{'*'*60}")

            for pcap_path in pcap_files:
                basename = os.path.basename(pcap_path)
                outfile = os.path.join(output_dir, basename)
                process_pcap(pcap_path, outfile)

        print(f"\n{'='*60}")
        print("All files processed successfully!")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\n[!] Unexpected error: {e}")
        cleanup_and_exit()


def main_single_dir():
    """
    Single input dir flow:
    1. Read raw PCAPs from INPUT_DIR
    2. Preprocess each into PROCESSED_DIR
    3. Split into TRAIN_DIR / TEST_DIR stratified by class (from filename)
    """
    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)

    if not os.path.isdir(INPUT_DIR):
        print(f"[!] Input directory not found: {INPUT_DIR}")
        sys.exit(1)

    pcap_files = sorted(
        glob.glob(os.path.join(INPUT_DIR, "*.pcap"))
        + glob.glob(os.path.join(INPUT_DIR, "*.pcapng"))
    )
    if not pcap_files:
        print(f"[!] No .pcap/.pcapng files found in {INPUT_DIR}")
        sys.exit(1)

    print(f"\n{'*'*60}")
    print(f"  Input:  {INPUT_DIR}  ({len(pcap_files)} files)")
    print(f"  Output: {TRAIN_DIR}/ and {TEST_DIR}/")
    print(f"{'*'*60}")

    try:
        # Step 1: Preprocess all PCAPs
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        for pcap_path in pcap_files:
            basename = os.path.basename(pcap_path)
            outfile = os.path.join(PROCESSED_DIR, basename)
            if os.path.exists(outfile):
                print(f"  [SKIP] Already processed: {basename}")
                continue
            try:
                process_pcap(pcap_path, outfile)
            except Exception as e:
                print(f"\n  [WARN] Failed to process {basename}: {e}, skipping.")

        # The train/test split now comes from the flows of the ground truth
        # (generate_dataset_with_flowid.py). Splitting by file is still
        # available through --split-only.
        print(f"\n{'='*60}")
        print("All files processed successfully!")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\n[!] Unexpected error: {e}")
        cleanup_and_exit()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", action="store_true",
                        help="Use legacy mode (separate train/test input dirs)")
    parser.add_argument("--split-only", action="store_true",
                        help="Skip preprocessing; only run the split that distributes "
                             "the already-processed pcaps between train and test")
    args = parser.parse_args()

    if args.split_only:
        split_processed_pcaps(PROCESSED_DIR, TRAIN_DIR, TEST_DIR)
    elif args.legacy:
        main()
    else:
        main_single_dir()
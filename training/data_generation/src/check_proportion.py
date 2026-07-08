import os
import glob
import subprocess
import configparser

def get_packet_count(directory):
    total_packets = 0
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return 0
    
    pcap_files = glob.glob(os.path.join(directory, '*.pcap')) + glob.glob(os.path.join(directory, '*.pcapng'))
    
    if not pcap_files:
        print(f"No pcap files found in {directory}.")
        return 0

    try:
        # We can pass all files to capinfos at once to make it extremely fast
        command = ['capinfos', '-T', '-r', '-m', '-c'] + pcap_files
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(',')
                if len(parts) >= 2 and parts[-1].isdigit():
                    total_packets += int(parts[-1])
    except subprocess.CalledProcessError as e:
        print(f"Error running capinfos on {directory}: {e.stderr}")
    except FileNotFoundError:
        print("capinfos command not found. Ensure wireshark/tshark is installed.")
        
    return total_packets

def main():
    config = configparser.ConfigParser()
    params_path = os.path.join(os.path.dirname(__file__), 'params.ini')
    if not os.path.exists(params_path):
        print(f"Error: params.ini not found at {params_path}")
        return

    config.read(params_path)
    config['DEFAULT']['training_root'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Handle the case where use_case is missing or misconfigured
    try:
        use_case = config['DEFAULT']['use_case']
        train_dir = config[use_case]['train_pcaps_path']
        test_dir = config[use_case]['test_pcaps_path']
    except KeyError as e:
        print(f"Missing configuration key in params.ini: {e}")
        return

    print(f"Checking packet proportions for use case: {use_case}")
    print(f"Train directory: {train_dir}")
    print(f"Test directory: {test_dir}\n")

    print("Running capinfos (this should be very fast)...")
    train_packets = get_packet_count(train_dir)
    test_packets = get_packet_count(test_dir)

    total = train_packets + test_packets

    print("-" * 50)
    if total == 0:
        print("No packets found in either directory.")
    else:
        print(f"Train Packets: {train_packets:,} ({train_packets/total:.2%})")
        print(f"Test Packets:  {test_packets:,} ({test_packets/total:.2%})")
        print(f"Total Packets: {total:,}")
    print("-" * 50)

if __name__ == '__main__':
    main()

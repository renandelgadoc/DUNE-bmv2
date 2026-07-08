import p4runtime_sh.shell as p4
import argparse

import sys
import logging
# Formatter
FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--p4info", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--grpc-port", required=True)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--inference-disabled", required=True)
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    inference_disabled = True if args.inference_disabled.lower() == 'true' else False

    # Set up logging
    log_level = getattr(logging, args.log_level.upper())

    logging.basicConfig(level=log_level, format=FORMAT)

    # Connecting to the switch
    logging.info('Establishing GRPC connection')
    p4.setup(
        device_id=args.device_id,
        grpc_addr=f"0.0.0.0:{args.grpc_port}",
        election_id=(0, 1),  # (high, low)
        config=p4.FwdPipeConfig(args.p4info, args.json),
        verbose=False,
    )
    logging.info('Switch program uploaded')

    if not inference_disabled:
        # Configure digest. In ensemble mode only the last switch in the chain
        # declares FlowDigest_t (the digest is fired once at the end of the
        # chain), so the other switches won't have it in their p4info. Skip
        # them gracefully instead of crashing the populate step.
        try:
            d = p4.DigestEntry("FlowDigest_t")
            d.ack_timeout_ns = 10 * 1000000000
            d.max_timeout_ns = 10 * 1000000000
            d.max_list_size = 10000
            d.insert()
            logging.info('Digest entry for FlowDigest_t inserted')
        except Exception as e:
            logging.info(f'FlowDigest_t not present on this switch, skipping digest setup ({e})')
    else:
        logging.info('No inference pipeline, digests not enabled')

    p4.teardown()


if __name__ == "__main__":
    main()

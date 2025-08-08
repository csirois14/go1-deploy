from policy_runner import PolicyRunner

from argparse import ArgumentParser
import time

SLEEP_TIME = 3  # seconds


def main(args):
    print(f"Sleeping for {SLEEP_TIME} seconds...")
    for i in range(SLEEP_TIME, 0, -1):
        print(f"{i}...", end="", flush=True)
        time.sleep(1)
    print()

    print("Starting Go1 Agent with the following parameters:")
    print(f"  Server: {args.server}")
    print(f"  Port: {args.port}")
    print(f"  External Policy: {'Enabled' if args.external_policy else 'Disabled'}")
    print(f"  Model Path: {args.model}")

    go_1 = PolicyRunner(
        path=args.model,
        server=args.server,
        port=args.port,
        joystick=not args.no_joystick,
        external_policy=args.external_policy,
    )
    go_1.init_pose()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--server", action="store_true", help="use a command server")
    parser.add_argument(
        "--external-policy", action="store_true", help="use external policy (actions from command server)"
    )
    parser.add_argument("-p", "--port", type=int, default=9292, help="port to receive commands from")
    parser.add_argument("-m", "--model", type=str, default="weights/base_policy.pt")
    arguments = parser.parse_args()

    # Validate arguments
    if arguments.external_policy and not arguments.server:
        parser.error("--external-policy requires --server")

    main(arguments)

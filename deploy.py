from policy_runner import PolicyRunner

from argparse import ArgumentParser
import time


def main(args):
    print("SLEEPING FOR 5 SECONDS")
    time.sleep(5)

    print("STARTING GO1 AGENT")
    go_1 = PolicyRunner(
        path=args.model, 
        server=args.server, 
        port=args.port, 
        joystick=not args.no_joystick,
        external_policy=args.external_policy
    )
    go_1.init_pose()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--server", action="store_true", help="use a command server")
    parser.add_argument("--no-joystick", dest="no_joystick", action="store_true", help="disable joystick")
    parser.add_argument("--external-policy", action="store_true", help="use external policy (actions from command server)")
    parser.add_argument("-p", "--port", type=int, default=9292, help="port to receive commands from")
    parser.add_argument(
        "-m", "--model", type=str, default="weights/rough_150000.pt"
    )  # 'weights/asym_full_model_10000.pt'
    arguments = parser.parse_args()
    
    # Validate arguments
    if arguments.external_policy and not arguments.server:
        parser.error("--external-policy requires --server")

    main(arguments)

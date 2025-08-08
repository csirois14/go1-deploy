#!/usr/bin/env python3
"""
Test script for external policy mode.

This script loads a policy, receives observations from the robot, computes actions using the policy,
and sends them back to the robot at 50 Hz.
"""

import socket
import struct
import time
import torch

# Constants matching policy_runner.py
ACTION_MSG_LEN = 50  # 2 bytes (code) + 12*4 bytes (actions)
CMD_CODE_ACTION = 2

SERVER_IP = "192.168.123.14"  # "192.168.123.14", or use "localhost" if `policy_runner` is on the same machine
PORT = 9292

POLICY_PATH = "weights/policy_good.pt"  # Path to the policy file


def load_policy(path):
    """Load the policy from a .pt file"""
    print(f"Loading policy from {path}...")
    policy = torch.jit.load(path, map_location="cpu")
    policy.eval()
    print("Policy loaded successfully!")
    return policy


def send_action_command(sock, actions):
    """
    Send action command to the robot.

    Args:
        sock: socket connection
        actions: list or array of 12 float values for joint actions
    """
    if len(actions) != 12:
        raise ValueError("Actions must contain exactly 12 values")

    # Pack: code (short) + 12 floats
    message = struct.pack("<h12f", CMD_CODE_ACTION, *actions)
    sock.send(message)


def receive_observations(sock):
    """Receive observations from the robot."""
    try:
        sock.settimeout(0.1)  # 100ms timeout

        # Read observation count
        count_data = sock.recv(1)
        if not count_data:
            return None

        obs_count = struct.unpack("B", count_data)[0]

        # Read observations (each observation is 4 bytes float)
        obs_bytes = obs_count * 4
        obs_data = b""
        while len(obs_data) < obs_bytes:
            chunk = sock.recv(obs_bytes - len(obs_data))
            if not chunk:
                return None
            obs_data += chunk

        # Unpack observations
        observations = struct.unpack(f"{obs_count}f", obs_data)
        return torch.tensor(observations, dtype=torch.float32)

    except socket.timeout:
        return None
    except Exception as e:
        print(f"Error receiving observations: {e}")
        return None


def main():
    policy_path = POLICY_PATH
    host = SERVER_IP
    port = PORT

    # Load the policy
    try:
        policy = load_policy(policy_path)

    except Exception as e:
        print(f"Failed to load policy: {e}")
        return

    print(f"Connecting to robot at {host}:{port}...")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
            print("Connected!")

            print("Starting external policy control at 50 Hz...")
            print("Press Ctrl+C to stop")

            step_count = 0
            start_time = time.time()
            last_obs_time = time.time()

            while True:
                loop_start = time.time()

                # Receive observations from robot
                observations = receive_observations(sock)

                # Note: The observations include the joystick state. You can leave them or overwrite them with your own values.

                if observations is not None:
                    last_obs_time = time.time()

                    # Compute actions using the policy
                    with torch.no_grad():
                        actions = policy(observations.unsqueeze(0))  # Add batch dimension
                        actions = actions.squeeze(0)  # Remove batch dimension

                    # Send actions to robot
                    actions_list = actions.detach().cpu().numpy().tolist()
                    send_action_command(sock, actions_list)

                    step_count += 1

                    # Print status every 2 seconds
                    if step_count % 100 == 0:
                        elapsed = time.time() - start_time
                        freq = step_count / elapsed if elapsed > 0 else 0
                        print(
                            f"Step {step_count}, Frequency: {freq:.1f} Hz, Actions: [{actions_list[0]:.3f}, {actions_list[1]:.3f}, {actions_list[2]:.3f}...]"
                        )

                else:
                    # No observations received - check if connection is still alive
                    if time.time() - last_obs_time > 1.0:
                        print("No observations received for 1 second - connection may be lost")
                        break

                # Maintain 50 Hz loop rate
                loop_time = time.time() - loop_start
                sleep_time = 0.02 - loop_time  # 50 Hz = 0.02 seconds
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except ConnectionRefusedError:
        print("Could not connect to robot. Make sure the robot is running with:")
        print("python deploy.py --server --external-policy")
    except KeyboardInterrupt:
        print(f"\nExternal policy control stopped after {step_count} steps")
    except Exception as e:
        print(f"Error during policy control: {e}")


if __name__ == "__main__":
    main()

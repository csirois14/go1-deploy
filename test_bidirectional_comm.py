#!/usr/bin/env python3
"""
Test script for bidirectional communication between policy runner and controller.
Run this after starting the policy runner with --server flag.
"""

import socket
import struct
import time
import numpy as np

SERVER_IP = "localhost"  # "192.168.123.14"  # Or use "localhost" for local testing
SERVER_PORT = 9292


def send_command(sock, x, y, r):
    """Send velocity command to the robot."""
    code = 1  # Command code
    data = struct.pack("<hfff", code, x, y, r)
    sock.sendall(data)
    print(f"Sent command: x={x:.2f}, y={y:.2f}, r={r:.2f}")


def send_action_command(sock, actions):
    """Send action command to the robot (for external policy mode)."""
    if len(actions) != 12:
        raise ValueError("Actions must contain exactly 12 values")
    
    code = 2  # Action command code
    data = struct.pack("<h12f", code, *actions)
    sock.sendall(data)
    print(f"Sent actions: [{', '.join([f'{a:.3f}' for a in actions[:3]])}...]")


def receive_observations(sock):
    """Receive observations from the robot."""
    try:
        sock.settimeout(0.1)  # 100ms timeout

        # Read observation count
        count_data = sock.recv(1)
        if len(count_data) != 1:
            return None

        obs_count = struct.unpack("B", count_data)[0]
        if obs_count == 0 or obs_count > 100:
            return None

        # Read observation data
        obs_data_size = obs_count * 4
        obs_data = b""
        while len(obs_data) < obs_data_size:
            chunk = sock.recv(obs_data_size - len(obs_data))
            if not chunk:
                return None
            obs_data += chunk

        observations = struct.unpack(f"{obs_count}f", obs_data)
        return np.array(observations)

    except (socket.timeout, socket.error, struct.error):
        return None
    finally:
        sock.settimeout(None)


def get_observation_dict(obs: np.array) -> dict:
    """Convert observation tensor to a dictionary for easier access."""
    obs_dict = {
        "base_lin_vel": obs[0:3].flatten(),
        "base_ang_vel": obs[3:6].flatten(),
        "projected_gravity": obs[6:9].flatten(),
        "velocity_commands": obs[9:12].flatten(),
        "joint_pos": obs[12:24].flatten(),
        "joint_vel": obs[24:36].flatten(),
        "actions": obs[36:48].flatten(),
    }
    return obs_dict


def main():
    print("Connecting to policy runner...")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((SERVER_IP, SERVER_PORT))
            print(f"Connected to {SERVER_IP}:{SERVER_PORT}")

            # Variables for frequency measurement
            obs_count = 0
            start_time = time.time()
            last_freq_print = start_time
            freq_window = 5.0  # Print frequency every 5 seconds

            # Test loop
            for i in range(50):  # Run for 50 iterations
                # Send a simple command
                x = 0.2 * np.sin(i * 0.1)  # Oscillating forward command
                y = 0.0
                r = 0.1 * np.cos(i * 0.1)  # Oscillating rotation

                # send_command(s, x, y, r)

                # Try to receive observations
                obs = receive_observations(s)
                if obs is not None:
                    obs_count += 1
                    current_time = time.time()
                    obs_dict = get_observation_dict(obs)

                    print(f"  Received {len(obs)} observations")

                    # Print some key observations
                    if len(obs) >= 45:
                        angular_vel = obs_dict["base_ang_vel"]
                        commands = obs_dict["velocity_commands"]
                        joint_angles = obs_dict["joint_pos"]
                        print(f"  Angular velocity: [{angular_vel[0]:.3f}, {angular_vel[1]:.3f}, {angular_vel[2]:.3f}]")
                        print(f"  Current commands: [{commands[0]:.3f}, {commands[1]:.3f}, {commands[2]:.3f}]")
                        print(f"  Joint angles: [{joint_angles[0]:.3f}, {joint_angles[1]:.3f}, {joint_angles[2]:.3f}]")
                else:
                    print("  No observations received")

                time.sleep(0.02)  # 50Hz update rate

            # Final frequency calculation
            final_time = time.time()
            total_elapsed = final_time - start_time
            if total_elapsed > 0 and obs_count > 0:
                final_frequency = obs_count / total_elapsed
                print(
                    f"\n*** Final observation frequency: {final_frequency:.2f} Hz ({obs_count} observations in {total_elapsed:.1f}s) ***"
                )

            # Send stop command
            send_command(s, 0.0, 0.0, 0.0)
            print("Test completed successfully!")

    except ConnectionRefusedError:
        print(f"Error: Could not connect to {SERVER_IP}:{SERVER_PORT}")
        print("Make sure the policy runner is running with --server flag")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

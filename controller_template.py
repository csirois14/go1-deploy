"""
This is the template of a python controller script to use with a server-enabled Agent.
"""

import struct
import socket
import time

import numpy as np
import cv2
import pyrealsense2 as rs


CONNECT_SERVER = True  # False for local tests, True for deployment


# ----------- DO NOT CHANGE THIS PART -----------

# The deploy.py script runs on the Jetson Nano at IP 192.168.123.14
# and listens on port 9292
# whereas this script runs on one of the two other Go1's Jetson Nano

SERVER_IP = "192.168.123.14"
SERVER_PORT = 9292

# Maximum duration of the task (seconds):
TIMEOUT = 180

# Minimum control loop duration:
MIN_LOOP_DURATION = 0.1


# Use this function to send commands to the robot:
def send(sock, x, y, r):
    """
    Send a command to the robot.

    :param sock: TCP socket
    :param x: forward velocity (between -1 and 1)
    :param y: side velocity (between -1 and 1)
    :param r: yaw rate (between -1 and 1)
    """
    data = struct.pack("<hfff", code, x, y, r)
    if sock is not None:
        sock.sendall(data)


def receive_observations(sock):
    """
    Receive observations from the robot.

    :param sock: TCP socket
    :return: numpy array of observations or None if no data available
    """
    if sock is None:
        return None

    try:
        # Set socket to non-blocking to avoid hanging
        sock.settimeout(0.001)  # 1ms timeout

        # Read the observation count (1 byte)
        count_data = sock.recv(1)
        if len(count_data) != 1:
            return None

        obs_count = struct.unpack("B", count_data)[0]
        if obs_count == 0 or obs_count > 100:  # Sanity check
            return None

        # Read the observation data (obs_count * 4 bytes for floats)
        obs_data_size = obs_count * 4
        obs_data = b""
        while len(obs_data) < obs_data_size:
            chunk = sock.recv(obs_data_size - len(obs_data))
            if not chunk:
                return None
            obs_data += chunk

        # Unpack the float data
        observations = struct.unpack(f"{obs_count}f", obs_data)
        return np.array(observations)

    except (socket.timeout, socket.error, struct.error):
        # No data available or error occurred
        return None
    finally:
        # Reset socket to blocking mode
        sock.settimeout(None)


# Fisheye camera (distortion_model: narrow_stereo):

image_width = 640
image_height = 480

# --------- CHANGE THIS PART (optional) ---------

pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == "RGB Camera":
        found_rgb = True
        break
if not found_rgb:
    print("Could not find a depth camera with color sensor")
    exit(0)

# Depht available FPS: up to 90Hz
config.enable_stream(rs.stream.depth, image_width, image_height, rs.format.z16, 30)
# RGB available FPS: 30Hz
config.enable_stream(rs.stream.color, image_width, image_height, rs.format.bgr8, 30)
# # Accelerometer available FPS: {63, 250}Hz
# config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
# # Gyroscope available FPS: {200,400}Hz
# config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

# Start streaming
pipeline.start(config)

# ----------- DO NOT CHANGE THIS PART -----------

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


arucoParams = cv2.aruco.DetectorParameters_create()
arucoParams.markerBorderBits = 1

RECORD = False
history = []

# ----------------- CONTROLLER -----------------

try:
    # We create a TCP socket to talk to the Jetson at IP 192.168.123.14, which runs our walking policy:

    print("Client connecting...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if CONNECT_SERVER:
            s.connect((SERVER_IP, SERVER_PORT))
            print("Connected.")
        else:
            s = None

        code = 1  # 1 for velocity commands

        task_complete = False
        start_time = time.time()
        previous_time_stamp = start_time

        # main control loop:
        while not task_complete and not time.time() - start_time > TIMEOUT:
            # avoid busy loops:
            now = time.time()
            if now - previous_time_stamp < MIN_LOOP_DURATION:
                time.sleep(MIN_LOOP_DURATION - (now - previous_time_stamp))

            # ---------- CHANGE THIS PART (optional) ----------

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            if RECORD:
                history.append((depth_frame, color_frame))

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # --------------- CHANGE THIS PART ---------------

            # --- Receive robot observations ---
            robot_observations = receive_observations(s)
            if robot_observations is not None:
                print(f"Received {len(robot_observations)} observations from robot")
                # You can now use robot_observations in your control logic
                # Example: robot_observations contains [angular_vel(3), gravity(3), commands(3), joint_pos(12), joint_vel(12), actions(12)]

                # Extract specific parts of observations if needed:
                # angular_vel = robot_observations[0:3]
                # projected_gravity = robot_observations[3:6]
                # current_commands = robot_observations[6:9]
                # joint_positions = robot_observations[9:21]
                # joint_velocities = robot_observations[21:33]
                # last_actions = robot_observations[33:45]

            # --- Detect markers ---

            # Markers detection:
            grey_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            (detected_corners, detected_ids, rejected) = cv2.aruco.detectMarkers(
                grey_frame, aruco_dict, parameters=arucoParams
            )

            print(f"Tags in FOV: {detected_ids}")

            # --- Compute control ---

            x_velocity = 0.0
            y_velocity = 0.0
            r_velocity = 0.0

            # You can now use robot_observations in your control computation
            # Example:
            # if robot_observations is not None:
            #     # Use joint positions to modify commands
            #     joint_positions = robot_observations[9:21]
            #     # ... your control logic here ...

            # --- Send control to the walking policy ---

            send(s, x_velocity, y_velocity, r_velocity)

        print(f"End of main loop.")

        if RECORD:
            import pickle as pkl

            with open("frames.pkl", "wb") as f:
                pkl.dump(frames, f)
finally:
    # Stop streaming
    pipeline.stop()

# BSD 2-Clause License

# Copyright (c) 2023, Bandi Jai Krishna

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""
Modified by Yann Bouteiller and Charles Sirois.

This version features a local server that takes in velocity commands.
If the server kwarg is True, joystick commands for X, Y, YAW are ignored and instead provided by a local server.
The port can be set using the port kwarg.
"""

import sys
import time
import math
import numpy as np
import struct
import torch

# import scipy
import csv

import io


import socket
import struct
from threading import Thread, Lock

from _old.actor_critic import ActorCritic
# from LinearKalmanFilter import LinearKFPositionVelocityEstimator
# from FastLinearKF import FastLinearKFPositionVelocityEstimator

sys.path.append("unitree_legged_sdk/lib/python/arm64")
sys.path.append("unitree_legged_sdk/lib/python/amd64")
import robot_interface as sdk

MSG_LEN = 14  # 1 short, 3 float (4 bytes each) - for velocity commands
ACTION_MSG_LEN = 50  # 1 short, 12 float (4 bytes each) - for action commands
START_IDLE_TIME = 1100  # reccommended: 1100

# Message codes
CMD_CODE_VELOCITY = 1  # Velocity command message
CMD_CODE_ACTION = 2  # Action command message

# Observation message format: 1 byte for obs_count + obs_count * 4 bytes for floats
MAX_OBS_COUNT = 100  # Maximum number of observations to send
OBS_MSG_HEADER_LEN = 1  # 1 byte for observation count


class CommandServer:
    def __init__(self, port):
        self._lock = Lock()
        self.__code = -1
        self.__x = 0
        self.__y = 0
        self.__r = 0
        self.__actions = torch.zeros(12, dtype=torch.float)  # Last received actions
        self._current_connection = None
        self._connection_lock = Lock()

        self._s_thread = Thread(target=self._server_thread, args=(port,), daemon=True)
        self._s_thread.start()

    def _server_thread(self, port):
        buffer = b""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", port))
            s.listen()
            c = True
            while c:
                conn, addr = s.accept()
                with conn:
                    print(f"Connected by {addr}")
                    # Store current connection for sending observations
                    with self._connection_lock:
                        self._current_connection = conn

                    try:
                        while True:
                            data = conn.recv(1024)
                            if not data:
                                # broken connection
                                with self._lock:
                                    self.__code, self.__x, self.__y, self.__r = -1, 0, 0, 0
                                break
                            buffer += data

                            # Process messages based on their length
                            while len(buffer) >= 2:  # At least enough for the code
                                # Peek at the code to determine message type
                                code = struct.unpack("<h", buffer[:2])[0]

                                if code == CMD_CODE_VELOCITY and len(buffer) >= MSG_LEN:
                                    # Process velocity command
                                    msg = buffer[:MSG_LEN]
                                    buffer = buffer[MSG_LEN:]
                                    code, x, y, r = struct.unpack("<hfff", msg)
                                    with self._lock:
                                        self.__code, self.__x, self.__y, self.__r = code, x, y, r

                                elif code == CMD_CODE_ACTION and len(buffer) >= ACTION_MSG_LEN:
                                    # Process action command
                                    msg = buffer[:ACTION_MSG_LEN]
                                    buffer = buffer[ACTION_MSG_LEN:]
                                    unpacked = struct.unpack("<h12f", msg)
                                    code = unpacked[0]
                                    actions = torch.tensor(unpacked[1:], dtype=torch.float)
                                    with self._lock:
                                        self.__code = code
                                        self.__actions = actions

                                elif code <= 0:
                                    # Handle exit codes
                                    if len(buffer) >= MSG_LEN:
                                        msg = buffer[:MSG_LEN]
                                        buffer = buffer[MSG_LEN:]
                                        code, x, y, r = struct.unpack("<hfff", msg)
                                        with self._lock:
                                            self.__code, self.__x, self.__y, self.__r = code, 0, 0, 0
                                        if code < 0:
                                            c = False
                                        break
                                    else:
                                        break
                                else:
                                    # Unknown message type or incomplete message
                                    break

                    finally:
                        # Clear connection when client disconnects
                        with self._connection_lock:
                            self._current_connection = None
                        print("Client disconnected")

    def get(self):
        with self._lock:
            x, y, r = self.__x, self.__y, self.__r
        return x, y, r

    def get_actions(self) -> torch.Tensor:
        """Get the last received actions for external policy mode"""
        with self._lock:
            return self.__actions.clone()

    def send_observations(self, observations):
        """
        Send observations back to the connected client.

        Args:
            observations: torch.Tensor or numpy array of observations
        """
        with self._connection_lock:
            if self._current_connection is None:
                return False  # No client connected

            try:
                # TODO: This was crashing last time, because the numpy version on the robot was too old. I tried a fix, but can't test it
                # Convert observations to numpy if it's a torch tensor
                if hasattr(observations, "detach"):
                    # PyTorch tensor - use manual conversion to avoid numpy integration issues
                    obs_array = observations.detach().cpu().tolist()
                    obs_array = np.array(obs_array, dtype=np.float32)

                elif hasattr(observations, "cpu"):
                    # PyTorch tensor without gradients
                    obs_array = observations.cpu().tolist()
                    obs_array = np.array(obs_array, dtype=np.float32)

                else:
                    # Already a numpy array or list
                    obs_array = np.array(observations, dtype=np.float32)

                # Flatten the array and limit to MAX_OBS_COUNT
                obs_flat = obs_array.flatten()
                if len(obs_flat) > MAX_OBS_COUNT:
                    obs_flat = obs_flat[:MAX_OBS_COUNT]

                # Create message: 1 byte for count + floats
                obs_count = len(obs_flat)
                msg = struct.pack("B", obs_count)  # 1 byte for count
                msg += struct.pack(f"{obs_count}f", *obs_flat)  # floats

                self._current_connection.sendall(msg)
                return True

            except (socket.error, struct.error, OSError) as e:
                print(f"Error sending observations: {e}")
                # Clear connection on error
                self._current_connection = None
                return False
            except Exception as e:
                print(f"Error converting observations to numpy: {e}")
                # Try alternative approach - send as list
                try:
                    if hasattr(observations, "tolist"):
                        obs_list = observations.tolist()
                    else:
                        obs_list = list(observations)

                    # Flatten and limit
                    def flatten_list(lst):
                        result = []
                        for item in lst:
                            if isinstance(item, (list, tuple)):
                                result.extend(flatten_list(item))
                            else:
                                result.append(float(item))
                        return result

                    obs_flat = flatten_list(obs_list)
                    if len(obs_flat) > MAX_OBS_COUNT:
                        obs_flat = obs_flat[:MAX_OBS_COUNT]

                    # Create message
                    obs_count = len(obs_flat)
                    msg = struct.pack("B", obs_count)
                    msg += struct.pack(f"{obs_count}f", *obs_flat)

                    self._current_connection.sendall(msg)
                    return True

                except Exception as e2:
                    print(f"Fallback method also failed: {e2}")
                    return False


class PolicyRunner:
    def __init__(self, path, server=False, port=9292, external_policy=False):
        self.server = server
        self.external_policy = external_policy

        # load the policy (skip if using external policy)
        if not external_policy:
            self.policy = self._load_policy(path)
        else:
            self.policy = None

        if server:
            self.command_server = CommandServer(port)
        else:
            self.command_server = None

        self.dt = 0.02
        self.num_actions = 12
        self.observation_history_length = 1
        self.num_obs = 45 * 1  # 44*5 #48
        self.unit_obs = 45

        self.num_privl_obs = 466  # self.num_obs #421 # num_obs
        self.device = "cpu"
        self.path = path  #'bp4/model_1750.pt'
        self.d = {
            "FR_0": 0,
            "FR_1": 1,
            "FR_2": 2,
            "FL_0": 3,
            "FL_1": 4,
            "FL_2": 5,
            "RR_0": 6,
            "RR_1": 7,
            "RR_2": 8,
            "RL_0": 9,
            "RL_1": 10,
            "RL_2": 11,
        }

        PosStopF = math.pow(10, 9)
        VelStopF = 16000.0
        HIGHLEVEL = 0xEE
        LOWLEVEL = 0xFF

        self.init = True
        self.motiontime = 0
        self.timestep = 0
        self.time = 0
        self.initialized = False
        self.init_log_data = False

        self.base_ang_vel_list = []
        self.dof_pos_list = []
        self.projected_gravity_list = []
        self.dof_vel_list = []

        self.finish_rec = 0
        self.leg_data_buffer = []
        self.csv_filename = "data/joint_angles.csv"

        #####################################################################
        self.euler = np.zeros(3)
        self.buf_idx = 0

        self.smoothing_length = 12
        self.deuler_history = np.zeros((self.smoothing_length, 3))
        self.dt_history = np.zeros((self.smoothing_length, 1))
        self.euler_prev = np.zeros(3)
        self.timuprev = time.time()

        self.omegaBody = np.zeros(3)
        self.accel = np.zeros(3)
        self.smoothing_ratio = 0.2

        # Old env
        # self.default_angles = [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1, -1.5, -0.1, 1, -1.5]

        self.default_angles = [
            0.1,
            -0.1,
            0.1,
            -0.1,
            0.8,
            0.8,
            1.0,
            1.0,
            -1.5,
            -1.5,
            -1.5,
            -1.5,
        ]

        self.default_angles_tensor = torch.tensor(
            self.default_angles,
            device=self.device,
            dtype=torch.float,
            requires_grad=False,
        )

        self.action_scale = 0.25
        self.actions: torch.Tensor = torch.zeros(
            self.num_actions, device=self.device, dtype=torch.float, requires_grad=False
        )
        self.obs: torch.Tensor = torch.zeros(self.num_obs, device=self.device, dtype=torch.float, requires_grad=False)
        self.obs_storage = torch.zeros(
            self.unit_obs * (self.observation_history_length - 1), device=self.device, dtype=torch.float
        )

        self.udp = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
        self.safe = sdk.Safety(sdk.LeggedType.Go1)

        self.cmd = sdk.LowCmd()
        self.state = sdk.LowState()
        self.udp.InitCmdData(self.cmd)

    def _load_policy(self, file_path):
        with open(file_path, "rb") as f:
            file_bytes = io.BytesIO(f.read())

        # jit load the policy
        policy = torch.jit.load(file_bytes)

        policy.to("cpu").eval()

        return policy

        # #! Old versions
        # actor_critic = ActorCritic(
        #     num_actor_obs=self.num_obs,
        #     num_critic_obs=self.num_privl_obs,
        #     num_actions=12,
        #     actor_hidden_dims=[512, 256, 128],
        #     critic_hidden_dims=[512, 256, 128],
        #     activation="elu",
        #     init_noise_std=1.0,
        # )
        # loaded_dict = torch.load(self.path, map_location=torch.device("cpu"))
        # actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        # actor_critic.eval()
        # self.policy = actor_critic.act_inference

    def get_commands(self):
        # Commands
        lx = struct.unpack("f", struct.pack("4B", *self.state.wirelessRemote[4:8]))
        ly = struct.unpack("f", struct.pack("4B", *self.state.wirelessRemote[20:24]))
        rx = struct.unpack("f", struct.pack("4B", *self.state.wirelessRemote[8:12]))
        # ry = struct.unpack('f', struct.pack('4B', *self.state.wirelessRemote[12:16]))

        if self.server:
            x, y, r = self.command_server.get()
            # clip out-of-bounds values:
            if abs(x) > 1:
                x /= abs(x)
            if abs(y) > 1:
                y /= abs(y)
            if abs(r) > 1:
                r /= abs(r)
            # interpret as joystick commands:
            forward = x * 0.6
            side = y * 0.5
            rotate = r * 0.8

        else:
            forward = ly[0] * 0.6
            side = -lx[0] * 0.5
            rotate = -rx[0] * 0.8

            if abs(forward) < 0.30:
                forward = 0
            if abs(side) < 0.2:
                side = 0
            if abs(rotate) < 0.4:
                rotate = 0

        return forward, side, rotate

    def get_observations(self):
        """Get the current observations from the robot's sensors.

        Observations:
        +---------------------------------------------------------+
        | Active Observation Terms in Group: 'policy' (shape: (45,)) |
        +-----------+---------------------------------+-----------+
        |   Index   | Name                            |   Shape   |
        +-----------+---------------------------------+-----------+
        |     0     | base_ang_vel                    |    (3,)   |
        |     1     | projected_gravity               |    (3,)   |
        |     2     | velocity_commands               |    (3,)   |
        |     3     | joint_pos                       |   (12,)   |
        |     4     | joint_vel                       |   (12,)   |
        |     5     | actions                         |   (12,)   |
        +-----------+---------------------------------+-----------+

        """
        # Get commands
        forward, side, rotate = self.get_commands()

        # Get sensor data and compute necessary quantities
        self.q = self.getJointPos()
        self.dq = self.getJointVelocity()
        self.quat = self.getQuaternion()
        self.omegaBody = self.getBodyAngularVel()  # self.state.imu.gyroscope
        self.contact_estimate = self.contactEstimator()
        self.aBody = self.getBodyAccel()
        # self.base_lin_vel = self.getBodyVel()  # self.state.velocity

        # vel_estimator = FastLinearKFPositionVelocityEstimator(q, dq, p, v, quat, omegaBody, contact_estimate, body_accel)
        # self.base_lin_vel = self.runKF()
        # self.lin_vel = self.aBody*0.001
        self.R = self.get_rotation_matrix_from_rpy(self.state.imu.rpy)
        self.gravity_vector = self.get_gravity_vector()
        self.pitch = self.state.imu.rpy[1]
        self.roll = self.state.imu.rpy[0]

        # Convert to numpy arrays for compatibility
        base_ang_vel_np = np.array(self.omegaBody, dtype=np.float32)
        dof_pos_np = np.array([m - n for m, n in zip(self.q, self.default_angles)], dtype=np.float32)
        projected_gravity_np = np.array(self.gravity_vector, dtype=np.float32)
        commands_np = np.array([forward, side, rotate], dtype=np.float32)
        dof_vel_np = np.array(self.dq, dtype=np.float32)
        actions_np = np.array(self.actions.detach().cpu().tolist(), dtype=np.float32)

        # # Debugging output
        # if self.timestep % 50 == 0:
        #     print(f"base_ang_vel: {base_ang_vel_np}")
        #     print(f"projected_gravity: {projected_gravity_np}")
        #     print(f"commands: {commands_np}")
        #     print(f"dof_pos: {dof_pos_np}")
        #     print(f"dof_vel: {dof_vel_np}")
        #     print(f"actions: {actions_np}")

        # Concatenate all observations as numpy array
        obs_np = np.concatenate(
            [base_ang_vel_np, projected_gravity_np, commands_np, dof_pos_np, dof_vel_np, actions_np], axis=0
        )

        # Clip observations
        obs_np = np.clip(obs_np, -100, 100)

        # Convert back to torch tensor for policy
        self.obs = torch.from_numpy(obs_np).to(self.device)

        current_obs = self.obs

        # print("obs shape : ", (self.obs).shape)

        if self.observation_history_length > 1:
            self.obs = torch.cat((self.obs, self.obs_storage), dim=-1)
            self.obs_storage[: -self.unit_obs] = self.obs_storage[self.unit_obs :].clone()
            self.obs_storage[-self.unit_obs :] = current_obs

    def init_pose(self):
        while self.init:
            self.pre_step()
            self.get_observations()
            self.motiontime = self.motiontime + 1

            if self.motiontime < 100:
                self.setJointValues(self.default_angles, kp=5, kd=1)

            else:
                self.setJointValues(self.default_angles, kp=50, kd=5)

            if self.motiontime > START_IDLE_TIME:
                self.init = False

            # TODO: debugging
            # if self.server and self.command_server:
            #     self.command_server.send_observations(self.obs)

            self.post_step()

        print("Starting")
        print("Logging data")
        self.init_log_data = True
        # self.time = 0
        try:
            while True:
                self.pre_step()
                self.finish_rec += self.state.wirelessRemote[2]  # L1 button is pressed
                # print("L1 button : ", self.state.wirelessRemote[2])
                # print("Finish rec : ", self.finish_rec)
                if self.finish_rec >= 2:
                    print("L1 button pressed, exiting data logging loop")
                    break

                self.step()

        finally:
            # When the loop is exited (button is pressed), write all collected data to the CSV file
            print("Finished logging data")
            with open(self.csv_filename, "w", newline="") as csv_file:
                csv_writer = csv.writer(csv_file)
                # Adjust the column headers based on your joint data
                headers = (
                    ["Timestamp"]
                    + [f"Joint {i + 1} Angle" for i in range(12)]
                    + [f"Desired Joint {i + 1} Angle" for i in range(12)]
                )
                csv_writer.writerow(headers)
                csv_writer.writerows(self.leg_data_buffer)
            print("Finished writing CSV file")

    def pre_step(self):
        self.udp.Recv()
        self.udp.GetRecv(self.state)

    def step(self):
        """
        Has to be called after init_pose
        calls pre_step for getting udp packets
        calls policy with obs, clips and scales actions and adds default pose before sending them to robot
        calls post_step
        """
        self.get_observations()

        # Get actions either from policy or from external command server
        if self.external_policy and self.server and self.command_server:
            # Use actions from external policy
            self.actions = self.command_server.get_actions()
        else:
            # Use local policy
            self.actions = self.policy(self.obs)

        actions = torch.clip(self.actions, -100, 100).to("cpu").detach()

        scaled_actions = actions * self.action_scale
        final_angles = scaled_actions + self.default_angles_tensor
        des_angles = scaled_actions + self.default_angles_tensor

        if self.init_log_data:
            self.leg_data_buffer.append([self.timestep] + self.q + des_angles.tolist())

        # Send observations to controller (if server mode and client is connected)
        if self.server and self.command_server:
            self.command_server.send_observations(self.obs)

        # Send joint commands
        self.setJointValues(angles=final_angles, kp=20, kd=0.5)

        self.post_step()

    def post_step(self):
        """
        Offers power protection, sends udp packets, maintains timing
        """
        self.safe.PowerProtect(self.cmd, self.state, 9)
        self.udp.SetSend(self.cmd)
        self.udp.Send()
        time.sleep(max(self.dt - (time.time() - self.time), 0))
        if self.timestep % 100 == 0:
            print(f"{self.timestep}| frq: {1 / (time.time() - self.time)} Hz")

        self.time = time.time()
        self.timestep = self.timestep + 1

    ###
    # Getters and Setters for the robot state
    ###
    def getJointVelocity(self):
        # Old order
        # velocity = [
        #     self.state.motorState[self.d["FL_0"]].dq,
        #     self.state.motorState[self.d["FL_1"]].dq,
        #     self.state.motorState[self.d["FL_2"]].dq,
        #     self.state.motorState[self.d["FR_0"]].dq,
        #     self.state.motorState[self.d["FR_1"]].dq,
        #     self.state.motorState[self.d["FR_2"]].dq,
        #     self.state.motorState[self.d["RL_0"]].dq,
        #     self.state.motorState[self.d["RL_1"]].dq,
        #     self.state.motorState[self.d["RL_2"]].dq,
        #     self.state.motorState[self.d["RR_0"]].dq,
        #     self.state.motorState[self.d["RR_1"]].dq,
        #     self.state.motorState[self.d["RR_2"]].dq,
        # ]
        # 'FL_0', 'FR_0', 'RL_0', 'RR_0', 'FL_1', 'FR_1', 'RL_1', 'RR_1', 'FL_2', 'FR_2', 'RL_2', 'RR_2'

        velocity = [
            self.state.motorState[self.d["FL_0"]].dq,
            self.state.motorState[self.d["FR_0"]].dq,
            self.state.motorState[self.d["RL_0"]].dq,
            self.state.motorState[self.d["RR_0"]].dq,
            self.state.motorState[self.d["FL_1"]].dq,
            self.state.motorState[self.d["FR_1"]].dq,
            self.state.motorState[self.d["RL_1"]].dq,
            self.state.motorState[self.d["RR_1"]].dq,
            self.state.motorState[self.d["FL_2"]].dq,
            self.state.motorState[self.d["FR_2"]].dq,
            self.state.motorState[self.d["RL_2"]].dq,
            self.state.motorState[self.d["RR_2"]].dq,
        ]
        return velocity

    def getJointPos(self):
        # # Old angles
        # current_angles = [
        #     self.state.motorState[self.d["FL_0"]].q,
        #     self.state.motorState[self.d["FL_1"]].q,
        #     self.state.motorState[self.d["FL_2"]].q,
        #     self.state.motorState[self.d["FR_0"]].q,
        #     self.state.motorState[self.d["FR_1"]].q,
        #     self.state.motorState[self.d["FR_2"]].q,
        #     self.state.motorState[self.d["RL_0"]].q,
        #     self.state.motorState[self.d["RL_1"]].q,
        #     self.state.motorState[self.d["RL_2"]].q,
        #     self.state.motorState[self.d["RR_0"]].q,
        #     self.state.motorState[self.d["RR_1"]].q,
        #     self.state.motorState[self.d["RR_2"]].q,
        # ]

        current_angles = [
            self.state.motorState[self.d["FL_0"]].q,
            self.state.motorState[self.d["FR_0"]].q,
            self.state.motorState[self.d["RL_0"]].q,
            self.state.motorState[self.d["RR_0"]].q,
            self.state.motorState[self.d["FL_1"]].q,
            self.state.motorState[self.d["FR_1"]].q,
            self.state.motorState[self.d["RL_1"]].q,
            self.state.motorState[self.d["RR_1"]].q,
            self.state.motorState[self.d["FL_2"]].q,
            self.state.motorState[self.d["FR_2"]].q,
            self.state.motorState[self.d["RL_2"]].q,
            self.state.motorState[self.d["RR_2"]].q,
        ]

        return current_angles

    def setJointValues(self, angles, kp, kd):
        """
        Angles order:
        'FL_0', 'FR_0', 'RL_0', 'RR_0', 'FL_1', 'FR_1', 'RL_1', 'RR_1', 'FL_2', 'FR_2', 'RL_2', 'RR_2'
        """
        angles_map = {
            "FL_0": 0,
            "FR_0": 1,
            "RL_0": 2,
            "RR_0": 3,
            "FL_1": 4,
            "FR_1": 5,
            "RL_1": 6,
            "RR_1": 7,
            "FL_2": 8,
            "FR_2": 9,
            "RL_2": 10,
            "RR_2": 11,
        }
        for each_joint, index in angles_map.items():
            self.cmd.motorCmd[self.d[each_joint]].q = angles[index]
            self.cmd.motorCmd[self.d[each_joint]].dq = 0
            self.cmd.motorCmd[self.d[each_joint]].Kp = kp
            self.cmd.motorCmd[self.d[each_joint]].Kd = kd
            self.cmd.motorCmd[self.d[each_joint]].tau = 0.0

    def getBodyAngularVel(self):
        # self.omegaBody = self.smoothing_ratio * np.mean(self.deuler_history / self.dt_history, axis=0) + (
        #             1 - self.smoothing_ratio) * self.omegaBody

        self.omegaBody = (
            self.smoothing_ratio * np.array(self.state.imu.gyroscope) + (1 - self.smoothing_ratio) * self.omegaBody
        )

        return self.omegaBody

    def contactEstimator(self):
        contact_estimate = []
        foot_force = np.array(self.state.footForce)

        # Vectorized comparison
        contact_estimate = np.where(foot_force > 10, 0.5, 0)

        return contact_estimate

    def getBodyVel(self):
        vel = np.array(self.state.velocity)
        return vel

    def getBodyAccel(self):
        # Empirical values found with NMPC controller
        x_offset = 0.14641
        y_offset = -0.03673
        alpha = 0.1

        if not self.initialized:
            self.accel = np.array(self.state.imu.accelerometer)
            self.accel[0] -= x_offset
            self.accel[1] -= y_offset
            self.initialized = True
        else:
            offset_input = np.array(self.state.imu.accelerometer)
            offset_input[0] -= x_offset
            offset_input[1] -= y_offset
            self.accel = alpha * offset_input + (1.0 - alpha) * self.accel
        return self.accel

    def getQuaternion(self):
        return np.array(self.state.imu.quaternion)

    def quaternionToRotationMatrix(self):
        # compute rBody
        R = np.zeros((3, 3))
        R[0, 0] = 1 - 2 * self.quat[2] ** 2 - 2 * self.quat[3] ** 2
        R[0, 1] = 2 * self.quat[1] * self.quat[2] - 2 * self.quat[0] * self.quat[3]
        R[0, 2] = 2 * self.quat[1] * self.quat[3] + 2 * self.quat[0] * self.quat[2]
        R[1, 0] = 2 * self.quat[1] * self.quat[2] + 2 * self.quat[0] * self.quat[3]
        R[1, 1] = 1 - 2 * self.quat[1] ** 2 - 2 * self.quat[3] ** 2
        R[1, 2] = 2 * self.quat[2] * self.quat[3] - 2 * self.quat[1] * self.quat[0]
        R[2, 0] = 2 * self.quat[1] * self.quat[3] - 2 * self.quat[2] * self.quat[0]
        R[2, 1] = 2 * self.quat[2] * self.quat[3] + 2 * self.quat[1] * self.quat[0]
        R[2, 2] = 1 - 2 * self.quat[1] ** 2 - 2 * self.quat[2] ** 2
        return R

    # def cheaterVelocityEstimator(self):
    #     # rBody = self.quaternionToRotationMatrix()
    #     # Rbod = rBody.T
    #     # aWorld = Rbod@self.aBody
    #     # a = aWorld + g
    #     self.base_lin_vel = self.aBody*self.dt

    def get_rotation_matrix_from_rpy(self, rpy):
        """
        Get rotation matrix from the given quaternion.
        Args:
            q (np.array[float[4]]): quaternion [w,x,y,z]
        Returns:
            np.array[float[3,3]]: rotation matrix.
        """
        r, p, y = rpy
        R_x = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])

        R_y = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])

        R_z = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])

        rot = np.dot(R_z, np.dot(R_y, R_x))
        return rot

    def get_gravity_vector(self):
        grav = np.dot(self.R.T, np.array([0, 0, -1]))
        return grav

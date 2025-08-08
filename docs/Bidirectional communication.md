# Bidirectional Communication Guide

This guide explains how to use the new bidirectional communication system between the policy runner and the controller template.

## Overview

The system now supports:
1. **Commands**: Controller → Policy Runner (velocity commands x, y, r)
2. **Observations**: Policy Runner → Controller (robot state observations)

## How It Works

### Policy Runner Side (deploy.py / policy_runner.py)

The policy runner now sends observations back to the connected controller every step when running in server mode:

```python
# In PolicyRunner.step():
if self.server and self.command_server:
    self.command_server.send_observations(self.obs)
```

### Controller Side (controller_template.py)

The controller template can now receive observations using the new `receive_observations()` function:

```python
robot_observations = receive_observations(s)
if robot_observations is not None:
    # Process the observations
    print(f"Received {len(robot_observations)} observations")
```

## Observation Format

The observations are sent as a numpy array with 45 elements in this order:

| Index | Component         | Size | Description                                       |
| ----- | ----------------- | ---- | ------------------------------------------------- |
| 0-2   | Angular Velocity  | 3    | Body angular velocity (rad/s)                     |
| 3-5   | Projected Gravity | 3    | Gravity vector in body frame                      |
| 6-8   | Commands          | 3    | Current velocity commands [forward, side, rotate] |
| 9-20  | Joint Positions   | 12   | Joint angles relative to default pose             |
| 21-32 | Joint Velocities  | 12   | Joint velocities                                  |
| 33-44 | Actions           | 12   | Last policy actions                               |

## Usage Example

```python
robot_observations = receive_observations(s)
if robot_observations is not None:
    # Extract specific components
    angular_vel = robot_observations[0:3]
    projected_gravity = robot_observations[3:6] 
    current_commands = robot_observations[6:9]
    joint_positions = robot_observations[9:21]
    joint_velocities = robot_observations[21:33]
    last_actions = robot_observations[33:45]
    
    # Use in your control logic
    if np.linalg.norm(angular_vel) > 2.0:
        # Robot is spinning too fast, reduce rotation command
        r_velocity = 0.0
    
    # Check if front legs are in a good position for stepping
    front_left_hip = joint_positions[3]  # FL_0
    front_right_hip = joint_positions[0]  # FR_0
    # ... your control logic ...
```

## Performance Notes

- Observations are sent every step (50Hz)
- Non-blocking socket operations prevent timing issues
- Maximum 100 observations per message to prevent buffer overflows
- Automatic fallback if no client is connected

## Testing

1. **Start the policy runner with server mode**:
   ```bash
   python deploy.py --server --port 9292
   ```

2. **Run your controller**:
   ```bash
   python controller_template.py
   ```

3. **Check console output** for observation reception messages

## Troubleshooting

- **No observations received**: Check that both ends are using the same port
- **Connection drops**: Ensure network stability between devices
- **Performance issues**: Consider reducing observation frequency if needed

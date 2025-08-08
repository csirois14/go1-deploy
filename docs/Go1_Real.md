# REAL_unitree_robots

## Startup and Shutdown

### Before start-up

Make sure that the robot is placed on the leveling ground before starting the machine. The robot's abdominal support pad should be flat on the ground. The body level is not tilted on the ground. The robot calf is fully stowed（As shown below）, make sure that the robot's thighs and calves are not pressed by the body, otherwise the robot may fail to boot.

![image](https://user-images.githubusercontent.com/11501425/202524427-6209a868-cc26-44bf-b461-c356fc10081e.png)

### Startup
After placing the robot according to the requirements in the “Before starting up” section, start the following steps: 
- short press the power switch once
- press and hold the power switch for more than 2 seconds to turn on the battery (when the battery is turned on, the indicator light is the green light
is always on and the indicator shows the current battery level). 

Then the robot will perform the poweron self-test. If the self-test is successful, the robot will stand up to the initial height of the body, and the
boot is successful. If the robot is not stand up during the above process, the robot fails the self-test. If the boot fails, the robot can't stand up. At this time, you need to check that the robot is positioned as described in the section Before start-up".

### Shutdown
Before shutting down, please make sure that the robot stands on the level of the ground, make sure that the robot is in Static Standing State (the height of the robot body is at the initial height after starting up, the body level, the joystick has no operation, the state when standing statically). 

- Press and hold the handle L2 button and then click the A button three times, the robot will then complete the squat, stand up and lie down; 
- hold down the handle L2 button, and then click the B button twice, the robot then completes prone (Damping), prone (undamped) action; 
- after the robot enters the prone (undamped) state, press the power switch once
- then press the power switch for more than 2 seconds to turn off battery. When the battery is turned off, the indicators are off.

## Go 1 Networking

![Go1-UdeM-Network - Cisco network diagram](https://user-images.githubusercontent.com/4822928/233205940-cb7b5582-8fb1-4800-8f94-9c0504daf865.png)

Chart can be edited here:
https://lucid.app/lucidchart/561b2143-03c2-4a20-b4b5-438ad761d31c/edit?viewport_loc=-136%2C54%2C1349%2C773%2CC7tQDR4fa0EQ&invitationId=inv_a8addd88-335f-40e1-a01c-63772035429a

## Connecting to the robots

### A1

#### WIFI

SSID: UnitreeRoboticsA1-494
PASS: 00000000

#### SSH
HOST: 192.168.123.161
USER: unitree
PASS: 123
OS: Ubuntu 16.04

### Go1

#### WIFI

SSID: UnitreeRoboticsGO1-XXX
PASS: 00000000

#### SSH

HOSTNAME: raspberrypi

IP: 192.168.123.161 / 192.168.12.1 

USER: pi / root

PASS: 123 / 123 (disabled?)
&nbsp;
  
&nbsp;

HOSTNAME: nano2gb

IP: 192.168.123.13

USER: unitree / root

PASS: 123 / (disabled) 
&nbsp;
  
&nbsp;

HOSTNAME: unitree-desktop

IP: 192.168.123.14 / 192.168.123.15

USER: unitree / root

PASS: 123 / (disabled)
&nbsp;
  
&nbsp;


OS: Ubuntu 20.04

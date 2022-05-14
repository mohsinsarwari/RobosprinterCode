import odrive
from odrive.enums import *
import numpy as np
import time

### CONNECT HIP MOTOR TO AXIS 0
### AND KNEE MOTOR TO AXIS 1


###### DESIRED PARAMS #######

mode = 0 # 0 Position | 1 Velocity

omega = 10

hip_amp = 24
hip_phase = np.pi


knee_amp = 24
knee_phase = np.pi

odrv0 = odrive.find_any()

try: # Reboot causes loss of connection, use try to supress errors
	odrv0.reboot()
except:
	pass
print("Rebooted [2/7]")
odrv0 = odrive.find_any() # Reconnect to the Odrive
print("Connected [3/7]")

odrv0.axis0.requested_state = AXIS_STATE_IDLE
odrv0.axis1.requested_state = AXIS_STATE_IDLE

time.sleep(0.5)

odrv0.axis0.controller.input_vel = 0
odrv0.axis1.controller.input_vel = 0

odrv0.axis0.requested_state = AXIS_STATE_SENSORLESS_CONTROL
odrv0.axis1.requested_state = AXIS_STATE_SENSORLESS_CONTROL

time.sleep(1)

dt = 0.01

time_passed = 0

if mode == 0:

	while time_passed < (8 * np.pi / omega):

		hip_pos = hip_amp * np.sin(omega * time_passed + hip_phase)
		knee_pos = knee_amp * np.sin(omega * time_passed + knee_phase)

		hip_motor_vel = (1/3.7) * hip_vel
		knee_motor_vel = (1/3.7) * knee_vel

		time_passed += dt

		odrv0.axis0.controller.input_vel = hip_motor_vel

		odrv0.axis1.controller.input_vel = knee_motor_vel


		print(knee_motor_vel)

		
		time.sleep(dt)

elif mode == 1:

	while time_passed < (8 * np.pi / omega):

		hip_vel = hip_amp * omega * np.cos(omega * time_passed + hip_phase)
		knee_vel = knee_amp * omega * np.cos(omega * time_passed + knee_phase)

		hip_motor_vel = (1/3.7) * hip_vel
		knee_motor_vel = (1/3.7) * knee_vel

		time_passed += dt

		odrv0.axis0.controller.input_vel = hip_motor_vel

		odrv0.axis1.controller.input_vel = knee_motor_vel


		print(knee_motor_vel)

		
		time.sleep(dt)

# period = 0.5
# for i in range(5):
# 	odrv0.axis0.controller.input_vel = 10
# 	time.sleep(period)
# 	odrv0.axis0.controller.input_vel = -10
# 	time.sleep(period)	

odrv0.axis0.requested_state = AXIS_STATE_IDLE
odrv0.axis1.requested_state = AXIS_STATE_IDLE


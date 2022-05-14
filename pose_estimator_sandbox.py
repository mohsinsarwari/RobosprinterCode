import cv2
import mediapipe as mp
import time
import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('good1.MOV')

#which way is the person going? 0 is left, 1 is right
run_dir = 1

shoulder_num = 11
hip_num=23
knee_num=25
ankle_num=27

direction = np.array([-1, 0])

if run_dir:
	shoulder_num += 1
	hip_num += 1
	knee_num += 1
	ankle_num += 1
	direction = np.array([1, 0])


pTime = 0
hip_angles = []
knee_angles = []
times = []
curr_time = 0

prev_pos = []

def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])
    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

while True:
    success, img = cap.read()
    if not success:
        break
    curr_time += 1/60
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    height, width, channels = imgRGB.shape
    print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    try:
        results.pose_landmarks.landmark
    except:
        continue
    # Right side of body landmarks
    shoulder_lm = results.pose_landmarks.landmark[shoulder_num]
    hip_lm = results.pose_landmarks.landmark[hip_num]
    knee_lm = results.pose_landmarks.landmark[knee_num]
    ankle_lm = results.pose_landmarks.landmark[ankle_num]
    # Normalize coordinates to pixel space, which maps more closely to real space
    # Can try to use MediaPipe 2.5D recollection to get 3D positions of running
    shoulder = np.array([shoulder_lm.x, shoulder_lm.y])
    hip = np.array([hip_lm.x, hip_lm.y])
    knee = np.array([knee_lm.x, knee_lm.y])
    ankle = np.array([ankle_lm.x, ankle_lm.y])
    # Angle between the torso and hip
    vector_1 = shoulder - hip
    vector_2 = knee - hip
    torso_unit_vector = vector_1 / np.linalg.norm(vector_1)
    thigh_unit_vector = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(torso_unit_vector, thigh_unit_vector)
    hip_angle = np.degrees(np.arccos(dot_product))
    if np.dot(thigh_unit_vector, direction) < 0:
    	hip_angle = 360 - hip_angle
    hip_angles.append(hip_angle)
    # Angle between thigh and calf
    vector_1 = knee - hip
    vector_2 = ankle - knee
    thigh_unit_vector = vector_1 / np.linalg.norm(vector_1)
    calf_unit_vector= vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(thigh_unit_vector, calf_unit_vector)
    knee_angle = 180 - np.degrees(np.arccos(dot_product))
    knee_angles.append(knee_angle)
    times.append(curr_time)
    for id_num, lm in enumerate(results.pose_landmarks.landmark):
        h, w,c = img.shape
        cx, cy = int(lm.x*w), int(lm.y*h)
        if id_num == knee_num or id_num == hip_num:
            prev_pos.append(lm)
            for pos in prev_pos:
                print(pos.x)
                cv2.circle(img, (int(pos.x*w), int(pos.y*h)), 5, (0,255,0), cv2.FILLED)
        else:
            cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
    # cv2.putText(img, str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
    #cv2.putText(img, "Hip Angle:" + str(hip_angle), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(200,200,200), 3)
    #cv2.putText(img, "Knee Angle:" + str(knee_angle), (50,100), cv2.FONT_HERSHEY_SIMPLEX,1,(200,200,200), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(0)


# for i in range(20):

# 	hip_outlier_idx = 0
# 	hip_outlier_avg = 0
# 	hip_outlier_dist = 0
# 	knee_outlier_idx = 0
# 	knee_outlier_dist = 0
# 	knee_outlier_avg = 0

# 	for i in np.arange(0, len(hip_angles)):

# 		if i == 0:
# 			hip_avg = hip_angles[1]
# 		elif i == len(hip_angles) - 1:
# 			hip_avg = hip_angles[len(hip_angles) - 2]
# 		else:
# 			hip_avg = (hip_angles[i-1] + hip_angles[i+1]) / 2

# 		hip_dist = abs(hip_angles[i] - hip_avg)

# 		if hip_dist > hip_outlier_dist:
# 			hip_outlier_idx = i
# 			hip_outlier_dist = hip_dist
# 			hip_outlier_avg = hip_avg

# 		if i == 0:
# 			knee_avg = knee_angles[1]
# 		elif i == len(knee_angles) - 1:
# 			knee_avg = knee_angles[len(knee_angles) - 2]
# 		else:
# 			knee_avg = (knee_angles[i-1] + knee_angles[i+1]) / 2

# 		knee_dist = abs(knee_angles[i] - knee_avg)

# 		if knee_dist > knee_outlier_dist:
# 			knee_outlier_idx = i
# 			knee_outlier_dist = knee_dist
# 			knee_outlier_avg = knee_avg

# 	hip_angles[hip_outlier_idx] = hip_outlier_avg

# 	knee_angles[knee_outlier_idx] = knee_outlier_avg


n=3
hip_smoothed = np.convolve(hip_angles, np.ones(n)/n, mode='valid')
knee_smoothed = np.convolve(knee_angles, np.ones(n)/n, mode='valid')
res_knee = fit_sin(times[(n//2):-(n//2)], knee_smoothed)
res_hip = fit_sin(times[(n//2):-(n//2)], hip_smoothed)

plt.scatter(times, hip_angles, s=10, label="hip_angles", color="b")
plt.scatter(times, knee_angles, s=10, label="knee_angles", color="r")
# plt.plot(times, res_knee["fitfunc"](np.array(times)), "r-", label="knee_func", linewidth=2)
# plt.plot(times, res_hip["fitfunc"](np.array(times)), "b-", label="hip_func", linewidth=2)
plt.title("Theta Profile")
plt.ylabel("theta (deg)")
plt.xlabel("time (s)")
plt.legend()
plt.show()
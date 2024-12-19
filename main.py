import cv2
import mediapipe as mp
import pandas as pd

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

video_source = 0  
cap = cv2.VideoCapture(video_source)

columns = ['Rep', 'Elbow_Left', 'Elbow_Right', 'Wrist_Left', 'Wrist_Right']
data = []
rep_count = 0

print("Press 's' to start recording reps and 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

        elbow_left_y = left_elbow.y
        elbow_right_y = right_elbow.y
        wrist_left_y = left_wrist.y
        wrist_right_y = right_wrist.y

        cv2.putText(frame, f"Elbow L: {elbow_left_y:.2f}, Elbow R: {elbow_right_y:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Wrist L: {wrist_left_y:.2f}, Wrist R: {wrist_right_y:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            rep_count += 1
            data.append([rep_count, elbow_left_y, elbow_right_y, wrist_left_y, wrist_right_y])
            print(f"Recorded Rep {rep_count}")

    cv2.imshow('Push-Up Tracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data, columns=columns)
df.to_csv('pushup_data.csv', index=False)
print("Data saved to 'pushup_data.csv'.")

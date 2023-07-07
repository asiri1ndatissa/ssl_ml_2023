import cv2
import csv
import os
import mediapipe as mp

def extract_keypoints(video_path,sign):
    # Extract the filename and extension from the video path
    video_name = os.path.basename(video_path)
    video_name_without_extension = os.path.splitext(video_name)[0]

    # Create the output CSV file path
    csv_file_path = f'/content/drive/MyDrive/Asiri/keyPoints/{sign}/{video_name_without_extension}.csv'
    print('csv_file_path',csv_file_path)
    # Load video
    cap = cv2.VideoCapture(video_path)

    # Initialize Mediapipe solutions
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    # Initialize CSV writer
    csv_file = open(csv_file_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'row_id', 'type', 'landmark_index', 'x', 'y', 'z'])

    frame_count = 0
    with mp_holistic.Holistic(static_image_mode=False) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the image to RGB and process it with Mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            if results.left_hand_landmarks is not None:
                for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                    row_id = f"{frame_count}-left_hand-{idx}"
                    csv_writer.writerow([frame_count, row_id, 'left_hand', idx, landmark.x, landmark.y, landmark.z])

            if results.left_hand_landmarks is None:
                for i in range(21):
                    row_id = f"{frame_count}-left_hand-{i}"
                    csv_writer.writerow([frame_count, row_id, 'left_hand', i, '', '', ''])

            # Extract right hand keypoints
            if results.right_hand_landmarks is not None:
                for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                    row_id = f"{frame_count}-right_hand-{idx}"
                    csv_writer.writerow([frame_count, row_id, 'right_hand', idx, landmark.x, landmark.y, landmark.z])

            if results.right_hand_landmarks is None:
                for i in range(21):
                    row_id = f"{frame_count}-right_hand-{i}"
                    csv_writer.writerow([frame_count, row_id, 'right_hand', i, '', '', ''])

            # Extract face keypoints
            if results.face_landmarks is not None:
                for idx, landmark in enumerate(results.face_landmarks.landmark):
                    row_id = f"{frame_count}-face-{idx}"
                    csv_writer.writerow([frame_count, row_id, 'face', idx, landmark.x, landmark.y, landmark.z])

            if results.face_landmarks is None:
                for i in range(468):
                    row_id = f"{frame_count}-face-{i}"
                    csv_writer.writerow([frame_count, row_id, 'face', i, '', '', ''])

            # Extract pose keypoints
            if results.pose_landmarks is not None:
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    row_id = f"{frame_count}-pose-{idx}"
                    csv_writer.writerow([frame_count, row_id, 'pose', idx, landmark.x, landmark.y, landmark.z])

            if results.pose_landmarks is None:
                for i in range(33):
                    row_id = f"{frame_count}-pose-{i}"
                    csv_writer.writerow([frame_count, row_id, 'pose', i, '', '', ''])

            frame_count += 1

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    # Release the video capture and close the CSV file
    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
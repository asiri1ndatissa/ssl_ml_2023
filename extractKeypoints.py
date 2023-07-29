import csv
import os
import mediapipe as mp
import cv2

def extract_keypoints(video_path,sign,signer_id):
    # Extract the filename and extension from the video path
    video_name = os.path.basename(video_path)
    video_name_without_extension = os.path.splitext(video_name)[0]
    # folder_path = f'{folder}/{sign}'

    # if not os.path.exists(folder_path):
    #   os.makedirs(folder_path)
    #   print("Folder created successfully.")

    # Create the output CSV file path
    # csv_file_path = f'{folder_path}/{video_name_without_extension}.csv'
    csv_file_path = 'test.csv'

    print('csv_file_path',csv_file_path)
    # Load video
    cap = cv2.VideoCapture(video_path)

    # Initialize Mediapipe solutions
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    # # write train file
    # train_csv = open(TRAIN_FILE, 'w', newline='')
    # train_csv_writer = csv.writer(train_csv)
    # train_csv_writer.writerow([csv_file_path,1,sign])

    # write train file

    # with open(TRAIN_FILE, 'a', newline='') as train_csv:
    #     train_csv_writer = csv.writer(train_csv)
    #     train_csv_writer.writerow([csv_file_path, signer_id, sign])

    # Initialize CSV writer
    csv_file = open(csv_file_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'row_id', 'type', 'landmark_index', 'x', 'y', 'z'])
    # SIGNER_COUNTER += 1

    frame_count = 0
    with mp_holistic.Holistic(static_image_mode=False) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the image to RGB and process it with Mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            # Check if both left hand and right hand landmarks are found
            if results.left_hand_landmarks is not None or results.right_hand_landmarks is not None:
              print('Found')
              # Extract face keypoints
              if results.face_landmarks is not None:
                  for idx, landmark in enumerate(results.face_landmarks.landmark):
                      row_id = f"{frame_count}-face-{idx}"
                      capped_x = min(max(landmark.x, -1.0), 1.0)
                      capped_y = min(max(landmark.y, -1.0), 1.0)
                      capped_z = min(max(landmark.z, -1.0), 1.0)
                      csv_writer.writerow([frame_count, row_id, 'face', idx, capped_x,  capped_y, capped_z])

              if results.face_landmarks is None:
                  for i in range(468):
                      row_id = f"{frame_count}-face-{i}"
                      csv_writer.writerow([frame_count, row_id, 'face', i, 0, 0, 0])

              if results.left_hand_landmarks is not None:
                  for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                      row_id = f"{frame_count}-left_hand-{idx}"
                      capped_x = min(max(landmark.x, -1.0), 1.0)
                      capped_y = min(max(landmark.y, -1.0), 1.0)
                      capped_z = min(max(landmark.z, -1.0), 1.0)
                      csv_writer.writerow([frame_count, row_id, 'left_hand', idx, capped_x,  capped_y, capped_z])

              if results.left_hand_landmarks is None:
                  for i in range(21):
                      row_id = f"{frame_count}-left_hand-{i}"
                      csv_writer.writerow([frame_count, row_id, 'left_hand', i, 0, 0, 0])

              # Extract pose keypoints
              if results.pose_landmarks is not None:
                  for idx, landmark in enumerate(results.pose_landmarks.landmark):
                      row_id = f"{frame_count}-pose-{idx}"
                      capped_x = min(max(landmark.x, -1.0), 1.0)
                      capped_y = min(max(landmark.y, -1.0), 1.0)
                      capped_z = min(max(landmark.z, -1.0), 1.0)
                      csv_writer.writerow([frame_count, row_id, 'pose', idx, capped_x,  capped_y, capped_z])

              if results.pose_landmarks is None:
                  for i in range(33):
                      row_id = f"{frame_count}-pose-{i}"
                      csv_writer.writerow([frame_count, row_id, 'pose', i, 0, 0, 0])

              # Extract right hand keypoints
              if results.right_hand_landmarks is not None:
                  for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                      row_id = f"{frame_count}-right_hand-{idx}"
                      capped_x = min(max(landmark.x, -1.0), 1.0)
                      capped_y = min(max(landmark.y, -1.0), 1.0)
                      capped_z = min(max(landmark.z, -1.0), 1.0)
                      csv_writer.writerow([frame_count, row_id, 'right_hand', idx, capped_x,  capped_y, capped_z])

              if results.right_hand_landmarks is None:
                  for i in range(21):
                      row_id = f"{frame_count}-right_hand-{i}"
                      csv_writer.writerow([frame_count, row_id, 'right_hand', i, 0, 0, 0])

            frame_count += 1

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    # Release the video capture and close the CSV file
    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()

extract_keypoints('සුභ උදෑසනක් 100.mp4',1,1)
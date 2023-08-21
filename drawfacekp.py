import cv2
import mediapipe as mp

# Initialize mediapipe holistic model
mp_face_mesh = mp.solutions.face_mesh

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

REYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    246, 161, 160, 159, 158, 157, 173,
][::2]
LEYE = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    466, 388, 387, 386, 385, 384, 398,
][::2]
NOSE=[
    1,2,98,327
]
SLIP = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    191, 80, 81, 82, 13, 312, 311, 310, 415,
]
POSE = [
    11,13,15,12,14,16,23,24,
]

def draw_specific_keypoints(image, keypoints, indices, color=(0, 255, 0), radius=3):
    for idx in indices:
        cv2.circle(image, keypoints[idx], radius, color, -1)

# Read the image
image = cv2.imread('ok.png')
# Convert the BGR image to RGB before processing
results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Extract keypoints
if results.face_landmarks:
    # Convert landmarks to a list of tuples
    keypoints = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in results.face_landmarks.landmark]

    # Draw only the right and left eye keypoints
    draw_specific_keypoints(image, keypoints, REYE, (0, 0, 255))  # Drawing right eye with red color
    draw_specific_keypoints(image, keypoints, LEYE, (0, 255, 0))  # Drawing left eye with green color
    draw_specific_keypoints(image, keypoints, NOSE, (0, 155, 0))  # Drawing left eye with green color
    draw_specific_keypoints(image, keypoints, SLIP, (0, 155, 255))  # Drawing left eye with green color

if results.pose_landmarks:
    keypoints = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in results.pose_landmarks.landmark]
    draw_specific_keypoints(image, keypoints, POSE, (0, 255, 0))  # Drawing left eye with green color




cv2.imshow("Keypoints Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

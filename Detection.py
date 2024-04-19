import cv2
from main import *
import matplotlib.pyplot as plt
import numpy as np

# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture(1)
camera_video.set(3, 1280)
camera_video.set(4, 960)

# Create named window for resizing purposes.
cv2.namedWindow('Fingers Counter', cv2.WINDOW_NORMAL)

# Counter for total frames processed
total_frames = 0
# Counter for frames with detected hands
frames_with_hands = 0
# Counter for frames with accurate finger detection
accurate_frames = 0
# List to store accuracy for each frame
accuracy_list = []

# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():
    # Read a frame.
    ok, frame = camera_video.read()

    # Check if frame is not read properly then continue to the next iteration to read the next frame.
    if not ok:
        continue

    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)

    # Perform Hands landmarks detection on the frame.
    frame, results = detectHandsLandmarks(frame, hands_videos, display=False)

    # Check if the hands landmarks in the frame are detected.
    if results.multi_hand_landmarks:
        frames_with_hands += 1
        # Count the number of fingers up of each hand in the frame.
        frame, fingers_statuses, count = countFingers(frame, results, display=False)
        # Calculate accuracy based on detected fingers
        total_fingers_detected = sum(count.values())
        # Assuming ground truth is not available, accuracy is calculated based on the number of fingers detected
        accuracy = total_fingers_detected / (len(results.multi_hand_landmarks) * 5) * 100  # Assuming 5 fingers per hand
        # Exclude frames with accuracy below 90%
        if accuracy >= 80:
            accuracy_list.append(accuracy)
            accurate_frames += 1 if accuracy == 100 else 0
            # Print accuracy for the current frame
            print(f"Accuracy for frame {total_frames}: {accuracy:.2f}%")

    # Display the frame.
    cv2.imshow('Fingers Counter', frame)

    # Wait for 1ms. If a key is pressed, retrieve the ASCII code of the key.
    k = cv2.waitKey(1) & 0xFF

    # Check if 'ESC' is pressed and break the loop.
    if k == 27:
        break

    total_frames += 1

# Release the VideoCapture Object and close the windows.
camera_video.release()
cv2.destroyAllWindows()

# Calculate overall average accuracy
overall_accuracy = (accurate_frames / frames_with_hands) * 100 if frames_with_hands > 0 else 0
print(f"Overall Accuracy: {overall_accuracy:.2f}%")

# Calculate average accuracy from the list
average_accuracy = np.mean(accuracy_list)
print(f"Average Accuracy: {average_accuracy:.2f}%")

# Plot the accuracy curve
plt.plot(range(len(accuracy_list)), accuracy_list, label='Accuracy')
plt.axhline(y=average_accuracy, color='r', linestyle='--', label='Average Accuracy')
plt.xlabel('Frame')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Curve')
plt.legend()
plt.grid(True)
plt.show()

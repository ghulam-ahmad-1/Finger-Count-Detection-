# Fingers Counter

This project leverages computer vision techniques to detect hand landmarks using webcam input, count the number of fingers shown, and evaluate detection accuracy. It utilizes the `Mediapipe` library for hand detection and `OpenCV` for video capture and processing.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Output](#output)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation

Ensure you have Python 3.x installed. Install the required libraries using pip:
```bash
pip install opencv-python mediapipe matplotlib numpy pygame
```

## Usage

1. Connect your webcam.
2. Run the `Detection.py` script:
   ```bash
   python Detection.py
   ```

## File Descriptions

### `Detection.py`

This script captures video from the webcam, processes each frame to detect hand landmarks, counts the fingers, and calculates detection accuracy.

#### Key Steps:

1. **Initialize Webcam and Settings:**
   ```python
   camera_video = cv2.VideoCapture(1)
   camera_video.set(3, 1280)
   camera_video.set(4, 960)
   ```

2. **Frame Processing:**
   - Capture and horizontally flip each frame.
   - Detect hand landmarks using the `detectHandsLandmarks` function.
   - Count fingers using the `countFingers` function.
   - Calculate and display accuracy per frame.

3. **Display Results:**
   ```python
   cv2.imshow('Fingers Counter', frame)
   ```

4. **Calculate Accuracy:**
   ```python
   overall_accuracy = (accurate_frames / frames_with_hands) * 100 if frames_with_hands > 0 else 0
   average_accuracy = np.mean(accuracy_list)
   ```

5. **Plot Accuracy Curve:**
   ```python
   plt.plot(range(len(accuracy_list)), accuracy_list, label='Accuracy')
   plt.axhline(y=average_accuracy, color='r', linestyle='--', label='Average Accuracy')
   plt.xlabel('Frame')
   plt.ylabel('Accuracy (%)')
   plt.title('Accuracy Curve')
   plt.legend()
   plt.grid(True)
   plt.show()
   ```

### `main.py`

This script contains essential functions for detecting hand landmarks and counting fingers.

#### Functions:

1. **`detectHandsLandmarks(image, hands, draw=True, display=True)`**
   - Detects hand landmarks in an image.
   - Optionally draws landmarks and displays images.

2. **`countFingers(image, results, draw=True, display=True)`**
   - Counts the number of fingers up for each hand in the image.
   - Optionally draws the total finger count and displays the output image.

## Output

- Real-time video feed displaying detected hand landmarks and finger counts.
- Frame-wise accuracy printed in the console.
- Overall and average accuracy displayed.
- Accuracy curve plotted using Matplotlib.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Mediapipe](https://google.github.io/mediapipe/)
- [OpenCV](https://opencv.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pygame](https://www.pygame.org/)

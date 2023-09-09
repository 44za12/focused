# Focused

A real-time application that uses a webcam to determine the user's concentration level.

## Description

This application processes a video stream from the user's webcam, detects the face, and calculates a concentration score based on several heuristics including:
- Eye Aspect Ratio (EAR)
- Eyebrow Position
- Mouth Openness
- Head Pose

If the concentration score falls below a certain threshold, the user is alerted with a notification. The application also keeps track of scores over time, allowing for the generation of a report to visualize concentration trends.

## Installation

### Prerequisites

- Python 3.x
- Webcam

### Steps

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## Usage

1. Start the application using `python main.py`.
2. Make sure your face is visible to the webcam.
3. The application will process the video stream and calculate a concentration score.
4. If the concentration score drops below the set threshold, a notification will alert the user.
5. Press 'q' to exit the application at any time.

## Reporting

To generate a report on the concentration scores, run:
```bash
python report.py
```
This will produce a visualization of the concentration scores over time.

## License

This project is open-source and available under the MIT License.

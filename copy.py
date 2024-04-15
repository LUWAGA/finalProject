# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_DATA')

# Scan the data folder to collect unique action labels
actions = set()
video_lengths = {}  # Dictionary to store video lengths

for filename in os.listdir(DATA_PATH):
    if filename.endswith('.mp4'):
        action = filename.split('.')[0]  # Extract the action label from the filename
        actions.add(action)

        # Get the duration of the video in frames
        video_path = os.path.join(DATA_PATH, filename)
        clip = VideoFileClip(video_path)
        duration = int(clip.duration * clip.fps)  # Duration in frames
        video_lengths[action] = duration
        clip.close()

# Convert the set of unique action labels to a NumPy array
actions = np.array(list(actions))

# 1 video worth of data
no_sequences = 1

print("Actions Detected:")
print(actions)

print("\nVideo Lengths (in frames):")
for action, length in video_lengths.items():
    print(f"{action}: {length} frames")
    
    
    
    
    
    
import os
import cv2
import numpy as np
import mediapipe as mp
from moviepy.editor import VideoFileClip

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_DATA')

# Scan the data folder to collect unique action labels
actions = set()
video_lengths = {}  # Dictionary to store video lengths

for filename in os.listdir(DATA_PATH):
    if filename.endswith('.mp4'):
        action = filename.split('.')[0]  # Extract the action label from the filename
        actions.add(action)

        # Get the duration of the video in frames
        video_path = os.path.join(DATA_PATH, filename)
        clip = VideoFileClip(video_path)
        duration = int(clip.duration * clip.fps)  # Duration in frames
        video_lengths[action] = duration
        clip.close()

# Convert the set of unique action labels to a NumPy array
actions = np.array(list(actions))

# Path to the folder containing video files
video_folder = 'MP_DATA'

# Get a list of all video file paths
video_paths = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.mp4')]

# Set mediapipe model
mp_holistic = mp.solutions.holistic

for video_path in video_paths:
    # Get the action label from the video file name
    action = os.path.basename(video_path).split('.')[0]

    # Initialize the video capture object
    cap = cv2.VideoCapture(video_path)

    # Check if the video capture was successful
    if not cap.isOpened():
        print(f"Error: Failed to open the video file {video_path}")
        continue

    # Get the video length (in frames) from the video_lengths dictionary
    sequence_length = video_lengths[action]

    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Create folder for action if it doesn't exist
        action_folder = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_folder):
            os.makedirs(action_folder)

        # Get the next available sequence folder name
        sequence_id = 0
        sequence_folder = os.path.join(action_folder, str(sequence_id))
        while os.path.exists(sequence_folder):
            sequence_id += 1
            sequence_folder = os.path.join(action_folder, str(sequence_id))

        # Create the new sequence folder
        os.makedirs(sequence_folder)

        # Check if all NumPy files already exist in the sequence folder
        existing_files = [f for f in os.listdir(sequence_folder) if f.endswith('.npy')]
        if len(existing_files) == sequence_length:
            print(f"Extraction already done for {action}. Skipping...")
            continue

        # Loop through video length aka sequence length
        for frame_num in range(sequence_length):
            # Read feed
            ret, frame = cap.read()

            # Check if the frame is empty
            if not ret:
                print(f"Error: Failed to read frame from the video capture {video_path}")
                break

            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            # Draw landmarks
            draw_styled_landmarks(image, results)

            # Apply wait logic
            if frame_num == 0:
                cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence_id), (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # Show to screen
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(500)

            else:
                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence_id), (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # Show to screen
                cv2.imshow('OpenCV Feed', image)

                # Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(sequence_folder, str(frame_num))
                np.save(npy_path, keypoints)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # Release video capture
    cap.release()

# Close OpenCV windows
cv2.destroyAllWindows()    
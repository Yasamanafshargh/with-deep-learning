import os
from pathlib import Path
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

"""
This program compares videos and finds similar videos using deep learning.
Even if the video is manipulated or changed, it will recognize it.
Optimizations have been made to improve performance.
"""

# Configure TensorFlow to limit GPU memory usage (optional)
# Uncomment the following lines if you experience GPU memory issues
# physical_devices = tf.config.list_physical_devices('GPU')
# if physical_devices:
#     try:
#         tf.config.experimental.set_memory_growth(physical_devices[0], True)
#     except:
#         pass

# Define paths to your videos - you can add as many paths as you want
folder_dir1 = 'D:\\Downloads\\videos\\Waterscape_5 and Animal_4.mp4'
folder_dir2 = 'D:\\Downloads\\videos\\Animal_4.mp4'
video_paths = [folder_dir1, folder_dir2]

# Base output directory for saving frames (optional)
base_output_folder = Path('D:\\Downloads\\video.frames.deep_learning')
base_output_folder.mkdir(parents=True, exist_ok=True)

# Load pre-trained MobileNetV2 model + higher level layers
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
model = base_model  # Since we set pooling='avg', the output is already a feature vector

def extract_and_process_frames(video_path, frame_rate=10):
    """
    Extract frames from videos at specified intervals and compute feature embeddings.

    Parameters:
    video_path (str): Path to the input video file.
    frame_rate (int): Process every nth frame.

    Returns:
    features (list): List of feature vectors extracted from frames.
    """
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    features = []

    frames_batch = []
    batch_size = 32  # Number of frames to process in a batch

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frames at specified intervals (e.g., every 10th frame)
        if frame_num % frame_rate == 0:
            # Resize frame to 224x224 as required by MobileNetV2
            frame_resized = cv2.resize(frame, (224, 224))
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames_batch.append(frame_rgb)

            # Process batch if it reaches batch_size
            if len(frames_batch) == batch_size:
                batch_features = process_batch(frames_batch)
                features.extend(batch_features)
                frames_batch = []  # Reset batch

        frame_num += 1

    # Process any remaining frames in the batch
    if frames_batch:
        batch_features = process_batch(frames_batch)
        features.extend(batch_features)

    cap.release()
    return features

def process_batch(frames_batch):
    """
    Process a batch of frames and extract features.

    Parameters:
    frames_batch (list): List of frames as NumPy arrays.

    Returns:
    batch_features (list): List of feature vectors extracted from the batch.
    """
    # Convert list of frames to NumPy array
    batch = np.array(frames_batch, dtype=np.float32)

    # Preprocess the batch
    batch_preprocessed = preprocess_input(batch)

    # Extract features
    batch_features = model.predict(batch_preprocessed, verbose=0)
    return batch_features

def process_videos(video_paths):
    video_features = {}

    with ThreadPoolExecutor() as executor:
        future_to_video = {executor.submit(extract_and_process_frames, video_path): video_path for video_path in video_paths}

        for future in as_completed(future_to_video):
            video_path = future_to_video[future]
            video_name = Path(video_path).stem
            try:
                features = future.result()
                video_features[video_name] = features
                print(f"Completed processing video: {video_name}")
            except Exception as e:
                print(f"Error processing video {video_name}: {e}")

    return video_features

def compare_videos(video_features, similarity_threshold=0.8):
    """
    Compare features between videos using cosine similarity.

    Parameters:
    video_features (dict): Dictionary containing features for each video.
    similarity_threshold (float): Threshold for cosine similarity to consider frames as similar.

    """
    video_names = list(video_features.keys())
    total_similarities = 0
    total_comparisons = 0

    for i in range(len(video_names)):
        for j in range(i + 1, len(video_names)):
            video1, video2 = video_names[i], video_names[j]
            features1, features2 = video_features[video1], video_features[video2]

            print(f"\nComparing {video1} to {video2}:")

            # Compute pairwise cosine similarity between all frame features
            similarities = cosine_similarity(features1, features2)

            # Count similarities above the threshold
            high_similarity_count = np.sum(similarities >= similarity_threshold)
            total_similarities += high_similarity_count
            total_comparisons += similarities.size

            print(f"Number of similar frame pairs: {high_similarity_count}")
            print(f"Total frame comparisons: {similarities.size}")

            # Calculate percentage of similar frames
            if similarities.size > 0:
                percentage = (high_similarity_count / similarities.size) * 100
                print(f"Similarity Percentage: {percentage:.2f}%")

                # Decision based on similarity percentage
                if percentage >= 30:
                    print("Conclusion: The videos are likely the same.")
                else:
                    print("Conclusion: The videos are likely different.")
            else:
                print("No frame comparisons were made.")

    # Overall conclusion (optional)
    if total_comparisons > 0:
        overall_percentage = (total_similarities / total_comparisons) * 100
        print(f"\nOverall Similarity Percentage: {overall_percentage:.2f}%")
    else:
        print("No comparisons were made across all videos.")

# Process videos and extract features
video_features = process_videos(video_paths)

# Compare videos based on extracted features
compare_videos(video_features)

<!DOCTYPE html>
<html>
<head>

</head>
<body>

<h1>Video Similarity Detection using Deep Learning</h1>

<p>This Python script compares videos to identify similarities using deep learning techniques. It leverages a pre-trained convolutional neural network to extract feature embeddings from video frames and compares them using cosine similarity. The script is optimized for performance and can recognize similarities even if the videos have been manipulated or altered.</p>

<h2>Table of Contents</h2>
<ul>
    <li><a href="#overview">Overview</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#requirements">Requirements</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#usage">Usage</a>
        <ul>
            <li><a href="#configuring-video-paths">Configuring Video Paths</a></li>
            <li><a href="#running-the-script">Running the Script</a></li>
        </ul>
    </li>
    <li><a href="#how-it-works">How It Works</a>
        <ul>
            <li><a href="#frame-extraction-and-feature-embedding">Frame Extraction and Feature Embedding</a></li>
            <li><a href="#video-comparison">Video Comparison</a></li>
        </ul>
    </li>
    <li><a href="#configuration-options">Configuration Options</a></li>
    <li><a href="#example-output">Example Output</a></li>
    <li><a href="#notes">Notes</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    <li><a href="#contact">Contact</a></li>
</ul>

<h2 id="overview">Overview</h2>
<p>The script processes multiple videos to extract features from their frames using a pre-trained MobileNetV2 model. It then compares the extracted features between videos to find similarities. The script is optimized for performance using batch processing and multithreading.</p>

<h2 id="features">Features</h2>
<ul>
    <li><strong>Deep Learning-based Comparison</strong>: Uses MobileNetV2, a convolutional neural network trained on ImageNet, to extract feature embeddings from video frames.</li>
    <li><strong>Robust to Manipulations</strong>: Capable of recognizing similarities even if videos have been edited or manipulated.</li>
    <li><strong>Optimized Performance</strong>: Utilizes batch processing and multithreading to speed up frame extraction and feature computation.</li>
    <li><strong>Customizable Parameters</strong>: Allows configuration of frame extraction rate, batch size, similarity threshold, and more.</li>
</ul>

<h2 id="requirements">Requirements</h2>
<ul>
    <li><strong>Python</strong> 3.6 or higher</li>
    <li><strong>TensorFlow</strong> 2.x</li>
    <li><strong>OpenCV</strong> (<code>cv2</code>)</li>
    <li><strong>NumPy</strong></li>
    <li><strong>scikit-learn</strong></li>
</ul>

<h2 id="installation">Installation</h2>
<ol>
    <li><strong>Clone the repository</strong> (if applicable) or download the script file.</li>
    <li><strong>Install the required Python packages</strong>:
        <pre><code>pip install tensorflow opencv-python numpy scikit-learn</code></pre>
        <p><strong>Note</strong>: For GPU support, ensure you have the appropriate GPU drivers and CUDA installed. TensorFlow 2.x includes GPU support in the standard <code>tensorflow</code> package.</p>
    </li>
</ol>

<h2 id="usage">Usage</h2>

<h3 id="configuring-video-paths">Configuring Video Paths</h3>
<ol>
    <li><strong>Specify the paths to your videos</strong> in the script:
        <pre><code># Define paths to your videos - you can add as many paths as you want
video_paths = [
    'path/to/your/first_video.mp4',
    'path/to/your/second_video.mp4',
    # Add more paths as needed
]</code></pre>
    </li>
    <li><strong>Optional</strong>: Adjust the <code>base_output_folder</code> if you wish to change the directory where frames are saved (currently, frames are not saved to disk).</li>
</ol>

<h3 id="running-the-script">Running the Script</h3>
<p>Run the script using the command line:</p>
<pre><code>python video_similarity.py</code></pre>
<p><strong>Note</strong>: Replace <code>video_similarity.py</code> with the name of your script file.</p>

<h2 id="how-it-works">How It Works</h2>

<h3 id="frame-extraction-and-feature-embedding">Frame Extraction and Feature Embedding</h3>
<ul>
    <li><strong>Frame Extraction</strong>: The script extracts frames from each video at specified intervals (default is every 10th frame).</li>
    <li><strong>Preprocessing</strong>: Each frame is resized to 224x224 pixels and converted from BGR to RGB color space.</li>
    <li><strong>Batch Processing</strong>: Frames are processed in batches (default batch size is 32) to improve performance.</li>
    <li><strong>Feature Extraction</strong>: The pre-trained MobileNetV2 model is used to extract feature embeddings from each frame. The model is configured to output a feature vector for each frame by setting <code>include_top=False</code> and <code>pooling='avg'</code>.</li>
</ul>

<h3 id="video-comparison">Video Comparison</h3>
<ul>
    <li><strong>Cosine Similarity</strong>: The feature embeddings of frames from different videos are compared using cosine similarity.</li>
    <li><strong>Similarity Threshold</strong>: A threshold (default is 0.8) is used to determine if frame pairs are considered similar.</li>
    <li><strong>Analysis</strong>: The script calculates the number of similar frame pairs and the total number of comparisons, computing the percentage of similar frames between videos.</li>
    <li><strong>Conclusion</strong>: Based on the similarity percentage (default criterion is 30% or higher), the script concludes whether the videos are likely the same or different.</li>
</ul>

<h2 id="configuration-options">Configuration Options</h2>
<p>You can customize several parameters in the script:</p>
<ul>
    <li><strong>Frame Rate</strong> (<code>frame_rate</code>): Determines how often frames are extracted from the video. Default is every 10th frame.
        <pre><code>def extract_and_process_frames(video_path, frame_rate=10):
    # ...</code></pre>
    </li>
    <li><strong>Batch Size</strong> (<code>batch_size</code>): Number of frames processed in a batch. Default is 32.
        <pre><code>batch_size = 32  # Number of frames to process in a batch</code></pre>
    </li>
    <li><strong>Similarity Threshold</strong> (<code>similarity_threshold</code>): The cosine similarity threshold to consider frames as similar. Default is 0.8.
        <pre><code>def compare_videos(video_features, similarity_threshold=0.8):
    # ...</code></pre>
    </li>
    <li><strong>Similarity Percentage Criterion</strong>: The percentage of similar frames required to conclude that videos are the same. Default is 30%.
        <pre><code>if percentage >= 30:
    print("Conclusion: The videos are likely the same.")
else:
    print("Conclusion: The videos are likely different.")</code></pre>
    </li>
    <li><strong>GPU Memory Configuration</strong>: If you experience GPU memory issues, you can uncomment the following lines to limit TensorFlow's GPU memory usage:
        <pre><code># Configure TensorFlow to limit GPU memory usage (optional)
# Uncomment the following lines if you experience GPU memory issues
# physical_devices = tf.config.list_physical_devices('GPU')
# if physical_devices:
#     try:
#         tf.config.experimental.set_memory_growth(physical_devices[0], True)
#     except:
#         pass</code></pre>
    </li>
</ul>

<h2 id="example-output">Example Output</h2>
<p>After running the script, you will see output similar to the following:</p>
<pre><code>Completed processing video: first_video
Completed processing video: second_video

Comparing first_video to second_video:
Number of similar frame pairs: 150
Total frame comparisons: 500
Similarity Percentage: 30.00%
Conclusion: The videos are likely the same.

Overall Similarity Percentage: 30.00%</code></pre>

<h2 id="notes">Notes</h2>
<ul>
    <li><strong>Performance</strong>: Processing videos can be computationally intensive, especially for longer videos or higher frame rates. Ensure your system has sufficient resources.</li>
    <li><strong>GPU Support</strong>: For better performance, it's recommended to run the script on a machine with a GPU.</li>
    <li><strong>Frame Storage</strong>: Currently, frames are processed in-memory and not saved to disk. If you wish to save frames, you can modify the script accordingly.</li>
    <li><strong>Video Formats</strong>: Ensure that the videos are in a format supported by OpenCV.</li>
    <li><strong>Error Handling</strong>: The script includes basic error handling to catch exceptions during video processing.</li>
</ul>

<h2 id="license">License</h2>
<p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>

<h2 id="acknowledgments">Acknowledgments</h2>
<ul>
    <li>The script utilizes the MobileNetV2 model from TensorFlow's Keras applications.</li>
    <li>Inspired by techniques in video processing and deep learning for content similarity detection.</li>
    <li>Thanks to the open-source community for providing the tools and libraries used in this project.</li>
</ul>

<h2 id="contact">Contact</h2>
<p>For any questions or suggestions, please contact <a href="mailto:yafsharghasemloo@gmail.com">yafsharghasemloo@gmail.com</a>.</p>

<hr>

<h1>Code</h1>
<p>For reference, here is the main code of the script:</p>
<pre><code>import os
from pathlib import Path
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
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
</code></pre>

<hr>

<p><em>Please ensure you have the necessary permissions to use the videos and comply with all applicable laws and regulations when processing video content.</em></p>

</body>
</html>

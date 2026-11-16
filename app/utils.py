import tensorflow as tf
from typing import List
from pathlib import Path
import cv2
import os 
from difflib import SequenceMatcher

# Repository root, so paths work no matter which folder the app is started from
ROOT = Path(__file__).resolve().parent.parent
# Define the vocabulary for character mapping
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

# Create a mapping from characters to integers
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")

# Create a mapping from integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_data(path):
    """Load one GRID clip (frames) and its transcript (as character ids) given the video's path."""
    file_name = Path(str(path)).stem
    video_path = ROOT / 'data' / 's1' / f'{file_name}.mpg'
    alignment_path = ROOT / 'data' / 'alignments' / 's1' / f'{file_name}.align'
    return load_video(str(video_path)), load_alignments(str(alignment_path))

def load_alignments(path:str) -> List[str]: 
    # Read the lines from the alignment file
    with open(path, 'r') as f: 
        lines = f.readlines() 
    
    # Extract the tokens from the lines, excluding 'sil' tokens
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
    
    # Convert the tokens to integers using the char_to_num mapping
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def load_video(path:str) -> List[float]: 
    # Open the video file
    cap = cv2.VideoCapture(path)
    frames = []
    # Iterate over each frame in the video
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        # Convert the frame to grayscale
        frame = tf.image.rgb_to_grayscale(frame)
        # Extract a region of interest from the frame
        frames.append(frame[190:236,80:220,:])
    cap.release()
    
    # Calculate the mean and standard deviation of the frames
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    
    # Normalize the frames using the mean and standard deviation
    return tf.cast((frames - mean), tf.float32) / std


def similar(a, b):
    """Character-level similarity between two strings (0 to 1)."""
    return SequenceMatcher(None, a, b).ratio()


def word_accuracy(real: str, predicted: str) -> float:
    """Share of the real sentence's words that the prediction gets right, in order."""
    real_words, pred_words = real.split(), predicted.split()
    if not real_words:
        return 0.0
    matched = sum(b.size for b in SequenceMatcher(None, real_words, pred_words).get_matching_blocks())
    return matched / len(real_words)

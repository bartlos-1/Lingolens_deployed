"""LingoLens demo: lip reading from silent video (LipNet-style model trained on one GRID speaker)."""
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so the helper modules import from any working directory
from modelutil import load_model  # noqa: E402
from utils import ROOT, load_data, num_to_char, similar, word_accuracy  # noqa: E402

VIDEO_DIR = ROOT / "data" / "s1"

st.set_page_config(page_title="LingoLens", layout="centered")
st.title("LingoLens")
st.write(
    "Lip reading from silent video. Pick a clip, and the model predicts the sentence from the speaker's lip "
    "movements alone. The model was trained on one speaker from the GRID corpus, so it only works on clips "
    "like these: one person, fixed camera, short fixed-grammar sentences such as \"bin blue at f two now\"."
)


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_model()


@st.cache_data(show_spinner=False)
def to_mp4(video_path: str) -> bytes:
    """Browsers can't play GRID's .mpg files, so convert to H.264 MP4 with the bundled ffmpeg."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "clip.mp4")
        subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-i", video_path,
             "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-an", out, "-y"],
            check=True,
        )
        return Path(out).read_bytes()


def to_gif(frames) -> bytes:
    """What the model sees: the normalised mouth crop, rescaled to 0-255 for display."""
    arr = frames.numpy().squeeze(-1)
    arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
    images = [Image.fromarray(f) for f in arr]
    buf = io.BytesIO()
    images[0].save(buf, format="GIF", save_all=True, append_images=images[1:], duration=100, loop=0)
    return buf.getvalue()


def clean(text: str) -> str:
    """Drop GRID's "sp" (short pause) marker, which is a label token, not a spoken word."""
    return " ".join(w for w in text.split() if w != "sp")


options = sorted(p.name for p in VIDEO_DIR.glob("*.mpg"))
selected = st.selectbox("Choose a video", options, index=None, placeholder="Select a video...")

if selected:
    video_path = VIDEO_DIR / selected
    st.video(to_mp4(str(video_path)))

    if st.button("Lip read", use_container_width=True, type="primary"):
        frames, annotations = load_data(video_path)
        real_text = clean(tf.strings.reduce_join(num_to_char(annotations)).numpy().decode("utf-8"))

        col1, col2 = st.columns(2)
        with col1:
            st.caption("What the model sees (mouth region, 75 frames)")
            st.image(to_gif(frames), width=300)
        with col2:
            with st.spinner("Lip reading..."):
                yhat = get_model().predict(tf.expand_dims(frames, axis=0), verbose=0)
                decoded = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
                predicted = clean(tf.strings.reduce_join(num_to_char(decoded)).numpy().decode("utf-8"))

            st.caption("Actual sentence")
            st.markdown(f"**{real_text}**")
            st.caption("Predicted sentence")
            st.markdown(f"**{predicted}**")
            st.caption("Words correct")
            st.markdown(f"**{word_accuracy(real_text, predicted):.0%}**  "
                        f"(character similarity {similar(real_text, predicted):.0%})")

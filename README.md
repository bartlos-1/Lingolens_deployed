# LingoLens: lip reading from video

LingoLens takes a silent video of someone speaking and predicts the sentence from their lip movements.
It's a university project from 2024, built with Tamerlan Ormanbayev. The model follows the
[LipNet](https://arxiv.org/abs/1611.01599) architecture (Assael et al., Oxford / DeepMind), and our
implementation started from a public LipNet tutorial in TensorFlow.

**Demo:** https://lingoread.streamlit.app/ (free hosting, so the first load can take a minute)

## How it works

1. **Preprocessing (OpenCV):** each video is split into 75 frames, converted to grayscale, cropped to a fixed mouth
   region, and normalised.
2. **Model (TensorFlow/Keras):** three 3D convolution layers learn the shape and motion of the lips across frames,
   followed by two bidirectional LSTM layers (128 units each, with dropout) that read the sequence in both directions.
   A dense softmax layer outputs one character per frame.
3. **Training:** CTC loss, which lets the model learn the alignment between frames and characters without
   frame-level labels. Adam optimiser, learning rate 1e-4, about 100 epochs.

## Data and results

- Trained on the [GRID audio-visual corpus](https://doi.org/10.5281/zenodo.3625687), which has short, fixed-grammar
  sentences such as "put red at G9 now". We used one speaker (speaker 1).
- On 100 test videos from that same speaker, the model got about 99% of words right.
- That number is from a single speaker with a small, fixed vocabulary, so it shows the pipeline works.
  It does not show that the model would work on new people or natural speech. Training on more speakers,
  with face detection and data augmentation, would be the next step.

## Running it

```bash
git clone https://github.com/bartlos-1/Lingolens_deployed.git
cd Lingolens_deployed
python3.11 -m venv .venv && source .venv/bin/activate   # TensorFlow 2.15 needs Python 3.9-3.11
pip install -r requirements.txt
streamlit run app/streamlitapp.py
```

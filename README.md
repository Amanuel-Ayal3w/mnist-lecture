# MNIST Lecture: Watching a Neural Net Learn to Read Digits

A single Jupyter notebook that trains a small neural network on the **MNIST** handwritten digits (0–9) using only **NumPy**, and shows what happens inside it, in the style of Andrej Karpathy's *micrograd*, *makemore* and CS231n lectures.

No PyTorch or TensorFlow: the forward pass, backpropagation and training loop are all written out by hand, so every step is visible.

## How to run it

### Option 1: on your computer

You need Python 3.9 or newer.

```bash
git clone git@github.com:Amanuel-Ayal3w/mnist-lecture.git
cd mnist-lecture

# create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# install the dependencies (numpy, matplotlib, jupyter)
pip install -r requirments.txt

# start Jupyter and open the notebook
jupyter notebook mnist_mlp_visualization_lecture.ipynb
```

Then choose **Run → Run All Cells** (or press `Shift + Enter` on each cell to go step by step).

### Option 2: Google Colab (nothing to install)

1. Go to [colab.research.google.com](https://colab.research.google.com).
2. Choose **File → Open notebook → GitHub**, and paste the URL of this repository.
3. Open `mnist_mlp_visualization_lecture.ipynb` and choose **Runtime → Run all**.

### Notes

- The first run **downloads MNIST (about 11 MB)** and saves it as `mnist.npz` next to the notebook, so you need an internet connection once.
- A full run takes **about 2 minutes** on a laptop CPU. No GPU needed.

## What you will get

The network is **784 → 128 (tanh) → 64 (tanh) → 10**, about 110,000 parameters, and reaches **about 97.5% accuracy** on the test set.

Along the way the notebook shows:

| Section | What you will see |
|---|---|
| **1. The data** | Example digits for every class, class counts, how images become 784-number vectors |
| **2. The model** | How a micrograd neuron `tanh(w·x + b)` becomes a matrix layer `tanh(X @ W + b)` |
| **3. Initialization** | Why the starting loss should be `-ln(1/10) = 2.303`; a careless init starts at ~11.6 and saturates the tanh neurons (the black-and-white saturation plot); the fix |
| **4. Backprop by hand** | Every gradient derived line by line, then checked against numerical gradients |
| **5. Training** | A minibatch gradient descent loop that records loss, gradients and weight snapshots |
| **6. Loss curve** | Raw, log-scale and smoothed loss; train vs validation; the "hockey stick" of a bad init |
| **7. Weights and biases** | Individual parameters moving during training, weight distributions before vs after, output biases per digit, the update:data ratio and gradient flow per layer |
| **8. What the network sees** | First-layer weights turning from noise into stroke detectors, the images that excite each neuron, class templates of a linear classifier, saliency maps |
| **9. Evaluation** | Train/val/test accuracy, predictions with probabilities, the most confident mistakes, confusion matrix, accuracy per digit |

The notebook ends with a summary and **exercises** (different learning rates, ReLU instead of tanh, smaller networks, and more).

## Files

| File | Description |
|---|---|
| `mnist_mlp_visualization_lecture.ipynb` | The lecture notebook |
| `requirments.txt` | Python dependencies |
| `mnist.npz` | Created automatically on first run (not in the repo) |

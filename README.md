# MLP-From-Scratch

Multilayer Perceptron (MLP) made from scratch. A MLP is a type of neural network that uses layers of connected nodes to learn patterns. It gets its name from having multiple layers — typically an input layer, one or more middle (hidden) layers, and an output layer. [More about MLP here](https://medium.com/data-science/multilayer-perceptron-explained-a-visual-guide-with-mini-2d-dataset-0ae8100c5d1c).

## Getting Started

These instructions will give you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

Requirements for the software to run

- [Git](https://help.github.com/en/articles/set-up-git)
- [Python](https://www.python.org/downloads/)
- [Jupyter](https://jupyter.org/install)

### Installing

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### How to run

All the training and analysis lives in the notebook. Open it and run every cell
(top to bottom). MNIST is downloaded automatically on the first run and cached
in `data/` (ignored by git).

```bash
jupyter notebook notebooks/experimentos.ipynb
# or open it in VS Code and select the .venv kernel, then "Run All"
```

The notebook validates the network on XOR, runs the gradient check, trains the
main model on MNIST (reaching ~97.8% test accuracy), compares configurations and
saves the plots to `results/`.

## Architecture

The final model is a fully-connected MLP:

```
input 784  ->  hidden 128 (ReLU)  ->  hidden 64 (ReLU)  ->  output 10 (softmax)
```

- **Two hidden layers (128 and 64 neurons).** This meets the "at least 2 hidden
  layers" requirement and gives the network enough capacity to reach >97% on
  MNIST without being slow or overfitting. The funnel shape (128 -> 64) gradually
  compresses the representation toward the 10 output classes.
- **ReLU on the hidden layers.** It is cheap to compute and avoids the vanishing
  gradient that saturating functions (sigmoid/tanh) suffer from, so training is
  faster and more stable. The activation is configurable, so tanh/sigmoid can be
  swapped in (see the experiment comparison).
- **Softmax + cross-entropy at the output.** Softmax turns the 10 logits into a
  probability distribution, and cross-entropy is the natural loss for
  classification. Their gradient combines into the clean form `(probs - y)`.
- **He weight initialization.** Weights are drawn from a normal with
  `std = sqrt(2 / fan_in)`, calibrated for ReLU. Initializing with zeros would
  make every neuron in a layer identical (symmetry) and the network would never
  learn; the random init breaks that symmetry.
- **Training:** mini-batch SGD, `lr = 0.1`, batch size 64, 15 epochs.

## Walkthrough

This is my step by step journey to complete this project.

### Roadmap

| Step | What was done                                                                 |
|------|-------------------------------------------------------------------------------|
| 1    | Setting up the environment (venv, requirements.txt, __init__.py)              |
| 2    | activations.py — ReLU, sigmoid, tanh, softmax + derivatives                   |
| 3    | cross-entropy + gradient combined with softmax                                |
| 4    | network.py — forward pass (arbitrary number of layers, weight initialization) |
| 5    | network.py — backpropagation (chain rule)                                     |
| 6    | optimizers.py — SGD (and optionally: Momentum/Adam)                           |
| 7    | Validation — XOR + numeric gradient check                                     |
| 8    | MNIST — data loading, preprocessing and mini-batch SGD training               |
| 9    | Results — training curves, experiment comparison and confusion matrix         |
| 10   | Final README — how to run, architecture and decisions & difficulties          |

### 1. Repo setup

In the first step, I created the repository structure with the necessary files and folders, set up an isolated virtual environment (`.venv`), installed the libraries and established a stable base to work from. I kept the dependencies minimal and intentional: **NumPy** for all the matrix math (the only library allowed for the core of the network), **Matplotlib** for the required loss/accuracy plots, and **scikit-learn** only for the valued extras (the confusion matrix), never for the network itself. I also added `mlp/__init__.py` so that `mlp/` becomes an importable Python package, which lets the notebook do `from mlp.network import MLP` cleanly.

### 2. Activation Functions (activations.py)

In the second step, I implemented the functions that introduce nonlinearity into the network (without them, stacking layers would be pointless — it would simply collapse into a single linear transformation) and their derivatives, which are needed for backpropagation. I implemented **ReLU**, **sigmoid** and **tanh** for the hidden layers, and **softmax** for the output.

Each derivative takes `z` (the pre-activation, before the function) rather than the activated output, because the chain rule in backprop needs `da/dz` evaluated exactly at `z`. The softmax includes a numerical-stability trick: I subtract the row maximum before the exponential, which avoids `exp()` overflow without changing the result (softmax is invariant to adding a constant). Notably, softmax has **no** separate derivative here — I deliberately combine it with the cross-entropy in the next step, because together they simplify into a much cleaner and more stable gradient. Finally, I exposed an `ACTIVATIONS` registry that maps a string to the `(function, derivative)` pair, which is what makes the activation **configurable** per layer.

### 3. Loss function (losses.py)

In the third step, I created the loss function to measure how far off the network is (cross-entropy) and the gradient of the loss with respect to the logits — the starting point for backpropagation. Cross-entropy measures the distance between the predicted distribution (softmax probabilities) and the true one-hot label: the more confident and correct the network is, the lower the loss; the more confident and wrong, the higher.

The key decision here was to **combine softmax and cross-entropy into a single gradient** instead of deriving each one separately. The combination cancels the ugly terms of the softmax Jacobian and simplifies to `(probs - y) / N`, which is both elegant and numerically stable. I kept the loss *value* (`cross_entropy`, used to monitor and plot training) separate from the loss *gradient* (`softmax_cross_entropy_grad`, used to actually train). I also added a small `1e-12` inside the `log` to avoid `log(0) = -inf`, and divide by the batch size `N` so the loss is a per-example average, which keeps the gradient magnitude independent of batch size.

### 4. Forward pass (network.py)

In the fourth step, I built the `MLP` class, which constructs a network with an arbitrary number of layers and performs the forward pass — passing the input through each layer until it produces the output logits. I follow a `(batch, features)` convention throughout (each row is one example), so a layer is just `z = a @ W + b` followed by the activation, fully vectorized over the batch.

The most important decision here was **weight initialization**. I used He initialization (`std = sqrt(2 / fan_in)`), which is calibrated for ReLU and keeps the signal variance stable across layers so values neither explode nor vanish. I explicitly did **not** initialize with zeros: if all weights start at zero, every neuron in a layer computes the same thing, receives the same gradient and evolves identically — the symmetry is never broken and the network cannot learn. The random init is what breaks that symmetry. During the forward pass I cache the intermediate `z` and `a` of each layer (backprop needs them for the chain rule), and the output layer returns **raw logits** — softmax is applied later in the loss (training) or in `predict` (inference), never twice.

### 5. Backpropagation (network.py)

In the fifth step, I implemented backpropagation: the gradient of the loss with respect to every weight and bias, computed by propagating the output error backwards from the output to the input. This is what actually allows the network to learn.

I defined `delta = dL/dz` (the gradient with respect to a layer's pre-activation). The backward pass starts from the last layer, where `delta` is exactly the combined softmax+CE gradient `(probs - y) / N` from step 3. Then, for each layer going back to front, I compute `dW = a_prev.T @ delta` (since `z = a_prev @ W + b`, the chain rule gives the input transposed times the output error) and `db = sum(delta)` over the batch (the bias is broadcast in the forward pass, so its gradient comes back summed). To move to the previous layer, I push the error back through the weights with `delta @ W.T` and multiply by the **derivative of the previous activation** evaluated at its cached `z` — forgetting that activation-derivative term is the classic bug that stops the loss from falling. I do not divide by `N` again, because the loss gradient was already averaged in step 3. I validate all of this numerically in step 7.

### 6. Optimizer (optimizers.py)

In the sixth step, I implemented the optimizers, which update the weights from the backpropagation gradients. The essential algorithm is **SGD** (`W <- W - lr * dW`), with a configurable learning rate. I gave my SGD an optional **momentum** term, which accumulates a velocity in the direction of recent descents to smooth the path and damp oscillations. I also implemented **Adam** (a valued optional), which adapts a per-parameter learning rate from running estimates of the first moment (mean of the gradients) and second moment (mean of the squares), with bias correction so the early steps are not underestimated; it usually converges faster and is less sensitive to the initial learning rate.

All optimizers share the same interface — `step(weights, biases, grads_W, grads_b)` — and update the parameters **in place**, since the lists they receive are the network's own weights. Having both SGD and Adam also gave me an extra row for the experiment comparison in step 9.

### 7. Validation: XOR + Gradient Check (notebooks/experimentos.ipynb)

In the seventh step, before scaling up to MNIST, I validated the whole network on the smallest non-linear problem: XOR. XOR is not linearly separable, so only a network with hidden layers can solve it — a good minimal test for the forward/backward/optimizer working together.

I also implemented a numeric gradient check, which is the part that actually proves backpropagation is correct. The idea is to approximate the gradient of each weight directly from the definition of a derivative, `(L(p + e) - L(p - e)) / (2e)`, and compare it against the analytic gradient produced by `backward`. If they match (difference below `1e-5`), the chain rule was implemented correctly.

The result was a maximum difference of `~6e-12` between the analytic and numeric gradients (well below the `1e-5` threshold), and the network learned XOR perfectly (final loss `~0.0003`, predictions `[0 1 1 0]`). With the math validated, I had the confidence to scale the same network to MNIST.

### 8. MNIST — Data & Training (notebooks/experimentos.ipynb)

In the eighth step, I scaled the validated network to the real task: classifying the 10 handwritten digits of MNIST. I load the dataset from the same `.npz` file keras uses (downloaded once and cached locally, ignored by git), then preprocess it: each 28x28 image is flattened into a 784-vector, pixels are normalized to `[0, 1]`, and the labels are one-hot encoded to match the softmax + cross-entropy output. Normalizing the inputs matters — feeding raw `0–255` values makes the pre-activations large and destabilizes training.

For training I wrote a mini-batch loop: every epoch the data is shuffled and split into batches of 64, and for each batch I run forward -> backward -> optimizer step. Shuffling each epoch is what makes the gradient descent *stochastic*, which helps the network escape bad local patterns and generalize better.

With the architecture `784 -> 128 -> 64 -> 10` (two ReLU hidden layers), plain SGD at `lr = 0.1`, batch size 64 and 15 epochs, the network reached **97.8% test accuracy** in about 8 seconds — comfortably above the 92% target.

### 9. Results (notebooks/experimentos.ipynb)

In the ninth step, I generated the plots and analysis. The training curves (`results/training_curves.png`) show the loss falling steadily while train and test accuracy rise together and stay close, which means the network is learning without heavy overfitting.

**Experiment comparison.** I compared five configurations, changing one factor at a time (10 epochs each, same seed):

| Config | Architecture | Activation | Optimizer | Test acc |
|--------|--------------|------------|-----------|----------|
| A (baseline) | 784-128-64-10 | ReLU | SGD lr=0.1 | 0.9764 |
| B | 784-128-64-10 | ReLU | SGD lr=0.01 | 0.9531 |
| C | 784-128-64-10 | tanh | SGD lr=0.1 | 0.9755 |
| D | 784-128-64-10 | ReLU | Adam lr=1e-3 | 0.9769 |
| E | 784-256-10 | ReLU | SGD lr=0.1 | 0.9772 |

The clearest takeaway is the **learning rate**: dropping it from 0.1 to 0.01 (config B) cost over 2 points of accuracy in the same number of epochs, because the smaller steps simply hadn't converged yet. ReLU and tanh performed almost the same here; Adam reached a similar accuracy to SGD but converged faster in the early epochs. A single wider hidden layer (E) was competitive with two narrower ones.

**Confusion matrix.** On the test set, the most confused pairs were `5 -> 3` (22 times), `7 -> 2`, `8 -> 3` and `9 -> 4`. These are all visually similar digits, so the mistakes are intuitive rather than random — a good sign that the network learned meaningful features. The matrix is saved at `results/confusion_matrix.png`.

## Decisions and difficulties

**1. What was the hardest technical decision I made, and why?**

For me, the hardest decision was in the planning phase: figuring out what I should
do and in which order to reach a result I would be happy with. Deciding which path
to follow and which techniques to use to make sure I got the outcome I expected was
harder than writing any single piece of the code itself.

**2. What did I try that did not work, and what did I learn from it?**

I tried training with a very high number of epochs, expecting a better model, but it
barely changed the result at all. I spent a good amount of time for almost no gain.
What I learned is that bigger numbers are not always better, past a certain point
the model has already converged, and adding more epochs just wastes time.

**3. If I were to start over, what would I do differently?**

I would not do anything differently. I think I followed a good flow and got the
result I was expecting, and for me that is already enough.
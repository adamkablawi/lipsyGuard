import json
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MODEL_PATH = REPO / "hw" / "build" / "model.keras"
STATS_PATH = REPO / "old" / "axis_normalization_stats.pkl"
DATA_PATH = HERE / "rawData.json"

# Loads all X data into an array and all Y data into a matching array
def load_windows():
    with open(DATA_PATH) as f:
        d = json.load(f)
    X = np.array([np.stack(w["x"], w["y"], w["z"], axis=-1) for w in d])
    Y = np.array([int(w["label"]) for w in d])
    return X, Y

# Loads all means and stds into 2 np arrays
def load_norm_stats():
    with open(STATS_PATH) as f:
        s = json.load(f)
    mean = np.array([np.stack(s['X']['mean'], s['Y']['mean'], s['Z']['mean'])])
    std = np.array([s["X"]["std"], s["Y"]["std"], s["Z"]["std"]])  
    return mean, std
    
# Takes a Conv layer and the batch normalization layer after it and returns a single conv
# Basically done for hardware simplification
# BN(y) = gamma·(y − mean) / sqrt(var + eps) + beta (4 fixed numbers stored in model)
def fold_bn(conv, bn):
    w, b = conv.get_weights()
    gamma, beta, mean, var = bn.get_weights()
    s = gamma / np.sqrt(var + bn.epsilon)
    w_f = w * s
    b_f = (b - mean) * s + beta
    return w_f, b_f

# Return an ordered list of ops describing the layers of the model
# If Conv1D op, fold and append, 3 layers passed (Conv1D, BN, ReLU)
# If MaxPooling1D, it's just pooling, 1 layer
# If GAP1D, it's just gap, 1 layer
# If dense layer, it's just dense, 1 layer
def extract_ops(model):
    ops = []
    layers = model.layers
    i = 0
    while i < len(layers):
        layer = layers[i]
        if isinstance(layer, tf.keras.layers.Conv1D):
            w_f, b_f = fold_bn(layer, layers[i+1])
            ops.append(("conv", w_f, b_f))
            i += 3
        elif isinstance(layer, tf.keras.layers.MaxPooling1D):
            ops.append(("pool",))
            i += 1
        elif isinstance(layer, tf.keras.layers.GlobalAveragePooling1D):
            ops.append(("gap",))
            i += 1
        elif isinstance(layer, tf.keras.layers.Dense):
            w, b = layer.get_weights()
            relu = layer.activation.__name__ == "relu"
            ops.append(("dense", w, b, relu))
            i += 1
        else:
            i += 1  # InputLayer, Dropout
    return ops

def conv1d_same(x, w, b):
    K = w.shape[0] # Kernel width
    L = x.shape[1] # Num time steps
    pad = K // 2   # How many 0s to add at each end
    xp = np.pad(x, ((0,0),(pad,pad),(0,0))) # adding 0s
    y = np.zeros((x.shape[0], L, w.shape[2]), dtype=np.float32)
    for k in range(K): # For every tap in the filter
        # Padded input shifted by k steps (always L time steps)
        # Multiplying each Cin length vector by tap[k]'s weights
        # Results in a vector of length Cout numbers per time step
        y += xp[:, k:k+L] @ w[k]
    return y + b # adds bias

# Pairs adjacent time steps and extracts the max from each
def maxpool2(x):
    N, L, C = x.shape
    y = np.max(x.reshape(N, L//2, 2, C), axis=2)
    return y

# Averages across all time steps to check how apparent 
# a feature was over the entire signal
def gap(x):
    y = np.mean(x, axis=1)
    return y

# Combines derived features through MM and bias
def dense(x, w, b, relu):
    y = x @ w + b
    if relu:
        y = np.maximum(y, 0)
    return y

def forward(ops, x):
    for op in ops:
        kind = op[0]
        if kind == "conv":
            x = conv1d_same(x, op[1], op[2])
            x = np.maximum(x, 0)
        elif kind == "pool":
            x = maxpool2(x)
        elif kind == "gap":
            x = gap(x)
        elif kind == "dense":
            x = dense(x, op[1], op[2], op[3])
    return x

def softmax(z):
    y = np.exp(z - np.max(z, axis=1, keepdims=True))
    y = y / np.sum(y, axis=1, keepdims=True)
    return y

def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    X, y = load_windows()
    mean, std = load_norm_stats()
    Xn = ((X - mean) / std).astype(np.float32)

    print("X:", X.shape, "y:", y.shape, "seizures:", y.sum())   # (275,200,3) (275,) 68

    ops = extract_ops(model)
    for op in ops:
        print(op[0], [a.shape for a in op[1:] if hasattr(a, "shape")])  # 10 ops

    mine = softmax(forward(ops, Xn))
    ref = model.predict(Xn, verbose=0)
    print("max |prob diff|:", np.abs(mine - ref).max())                  # ~1e-6
    print("label agreement:", (mine.argmax(1) == ref.argmax(1)).mean())  # 1.0

if __name__ == "__main__":
    main()
    
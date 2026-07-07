---
title: Image Classifier
emoji: 🖼️
colorFrom: green
colorTo: gray
sdk: docker
pinned: false
---

# Image Classifier

A learning project that reinforces machine learning fundamentals by building dense and CNN models from scratch using Numpy only. 

The project also includes a web interface where you can upload an image and have the model classify it. 

**Features implemented from scratch:**
- Batch normalization (per-feature, with running stats for inference)
- Dropout regularization (`keep_prob = 0.8`)
- Adam optimizer with bias correction
- Mini-batch gradient descent
- He weight initialization

### CNN Model
*(In progress)*

---

## Project Structure

```
image-classifier/
├── notebooks/
    ├── testing.ipynb      # For testing
├── utils/
│   ├── activations.py     # ReLU, Softmax, CCE loss
│   ├── layers.py          # Forward/backward pass, BN, dropout, Adam
│   ├── train.py           # dense_model class (fit / predict)
│   ├── data_utils.py      # Data loading and preprocessing
│   └── evaluate.py        # Accuracy, confusion matrix
    └── README.md          # formulas, derivation, and coding plan
├── train/
│   └── weights.npz        # Saved model weights (git-ignored)
└── README.md
```

---

## Usage

**Train and save weights:**
```bash
git clone https://github.com/henrytran07/image-classifier.git
cd image-classifier
python -m utils.dense_model
```

## Demo

🌐 **Live Demo:** [henrytran07-image-classifier.hf.space](https://henrytran07-image-classifier.hf.space/)

---

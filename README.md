## Image Classifier 

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
├── utils/
│   ├── activations.py     # ReLU, Softmax, CCE loss
│   ├── layers.py          # Forward/backward pass, BN, dropout, Adam
│   ├── train.py           # dense_model class (fit / predict)
│   ├── data_utils.py      # Data loading and preprocessing
│   └── evaluate.py        # Accuracy, confusion matrix
├── train/
│   └── weights.npz        # Saved model weights (git-ignored)
└── README.md
```

---

## Usage

**Train and save weights:**
```bash
cd image-classifier
python -m utils.dense_model
```

**Run the web app:**
*(Coming soon)*

---
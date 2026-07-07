from fastapi import FastAPI, Request, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import json, os, io, numpy as np
from utils.activations import z_score_normalization
from utils.train import dense_model

LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

base_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base_dir, "train", "training_history.json"), "r") as f:
    train_dict = json.load(f)

data = np.load(os.path.join(base_dir, "train", "weights.npz"))
linear_params = {k: data[k] for k in ["W1","b1","W2","b2","W3","b3","W4","b4"]}
bn_params     = {k: data[k] for k in ["gamma1","beta1","gamma2","beta2","gamma3","beta3"]}
bn_running    = {k: data[k] for k in ["running_mean1","running_var1",
                                       "running_mean2","running_var2",
                                       "running_mean3","running_var3"]}
model = dense_model(linear_params, bn_params, bn_running)

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/assets", StaticFiles(directory="assets"), name="static")
app.mount("/train", StaticFiles(directory="train"), name="train")

@app.get("/", include_in_schema=False)
def home(request: Request):
    return templates.TemplateResponse(request, "home.html")

@app.get("/dense_model", include_in_schema=False)
def dense_model_page(request: Request):
    return templates.TemplateResponse(request, "dense_model.html", {
        "training_history": train_dict,
        "loss_graph_url": "/train/cce_loss.png",
        "accuracy_url": "/train/accuracy.png"
    })

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("L")
    img = img.resize((28, 28))
    x = np.array(img, dtype=np.float32)
    x_norm = z_score_normalization(x)
    x_norm = x_norm.reshape(-1, 1)
    pred = int(model.predict(x_norm).flatten()[0])
    return {"label": LABELS[pred]}
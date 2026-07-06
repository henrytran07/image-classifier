from fastapi import FastAPI, Request 
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import json 
import os 

base_dir = os.path.dirname(os.path.abspath(__file__))
path_json = os.path.join(base_dir, "train", "training_history.json")
loss_graph_dir = os.path.join(base_dir, "train", "cce_loss.png")
accuracy_dir = os.path.join(base_dir, "train", "accuracy.png")

train_dict = {}
with open(path_json, 'r') as f: 
    train_dict = json.load(f)

app = FastAPI() 
templates = Jinja2Templates(directory="templates")
app.mount("/assets", StaticFiles(directory="assets"), name='static')
app.mount("/train", StaticFiles(directory="train"), name="train")

@app.get("/", include_in_schema=False)
def home(request: Request): 
    return templates.TemplateResponse(request, "home.html")

@app.get("/dense_model", include_in_schema=False)
def dense_model(request: Request): 
    return templates.TemplateResponse(request, "dense_model.html", {
        "training_history": train_dict,
        "loss_graph_url": "/train/cce_loss.png",
        "accuracy_url": "/train/accuracy.png"
    })
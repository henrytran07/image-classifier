
from .train import * 
from .data_utils import * 
from .evaluate import * 
import numpy as np 
import json
import os 
import matplotlib.pyplot as plt
if __name__ == '__main__': 
    """
        Used for training 
    """
    # x_train_dm, y_train, x_val_dm, y_val, x_test_dm, y_test = load_dense_model_ds() 
    # model = dense_model_training() 
    # model.fit(x_train_dm, y_train, x_val_dm, y_val)

    """
        For regular use 
    """
    # x_train_dm, y_train, x_val_dm, y_val, x_test_dm, y_test = load_dense_model_ds()
    # BASE_DIR = Path(__file__).resolve().parents[1]
    # data = np.load(BASE_DIR / "train" / "weights.npz")
    # linear_params = {k: data[k] for k in ["W1","b1","W2","b2","W3","b3","W4","b4"]}
    # bn_params     = {k: data[k] for k in ["gamma1","beta1","gamma2","beta2","gamma3","beta3"]}
    # bn_running    = {k: data[k] for k in ["running_mean1","running_var1",
    #                                     "running_mean2","running_var2",
    #                                     "running_mean3","running_var3"]}
    # model = dense_model(linear_params, bn_params, bn_running)
    # yhat = model.predict(x_test_dm)
    # a = accuracy(yhat, y_test)
    # cm = confusion_matrix(yhat, y_test)
    # print(yhat)
    # print(f"Accuracy: {a}")
    # print(cm)

    """
        Generate loss graph 
    """
    # base_dir = os.path.dirname(os.path.abspath(__file__))
    # graph_dir = os.path.join(base_dir, "..", "train")
    # train_dir = os.path.join(base_dir, "..", "train" , "training_history.json")
    # train_dict = {}
    # with open(train_dir, "r") as f: 
    #     train_dict = json.load(f)

    # train_loss_data = np.array([train_dict["Epoch" + str(i)]['train_loss'] for i in range(1, 21)])
    # val_loss_data = np.array([train_dict["Epoch" + str(i)]['val_loss'] for i in range(1, 21)])
    # accuracy_data = np.array([train_dict["Epoch" + str(i)]['val_accuracy'] for i in range(1, 21)] )
    # epochs = np.arange(1, 21)    
    # plt.figure(figsize=(8, 5))
    # plt.plot(epochs, train_loss_data, label="Train Loss", marker="o")
    # plt.plot(epochs, val_loss_data, label="Val Loss", marker="x")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Training History")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # os.makedirs(graph_dir, exist_ok=True)
    # plt.savefig(f"{graph_dir}/cce_loss.png", dpi=150, bbox_inches="tight")
    # plt.show()

    # plt.figure(figsize=(8, 5))
    # plt.plot(epochs, accuracy_data, label="Accuracy", marker="o")
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.title("Accuracy History")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # os.makedirs(graph_dir, exist_ok=True)
    # plt.savefig(f"{graph_dir}/accuracy.png", dpi=150, bbox_inches="tight")
    # plt.show()

    """
        Save some test image
    """
    LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    _, _, _, _, x_test_dm, y_test = load_dense_model_ds() 
    img = [x_test_dm[:, i].reshape(28, 28, 1) for i in range(10)]
    ground_truth = [LABELS[y_test[0, i]] for i in range(10)]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, "..", "test")
    os.makedirs(img_dir, exist_ok=True)

    for i in range(10): 
        plt.figure(figsize=(8, 5))
        plt.imshow(img[i].squeeze(), cmap="gray") 
        plt.title(f"Label: {ground_truth[i]}")
        plt.axis("off")
        plt.savefig(f"{img_dir}/img{i}.png", dpi=150, bbox_inches="tight")
        plt.close() 



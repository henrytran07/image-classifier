
from .train import * 
from .data_utils import * 
from .evaluate import * 
import numpy as np 
from pathlib import Path
if __name__ == '__main__': 
    x_train_dm, y_train, x_val_dm, y_val, x_test_dm, y_test = load_dense_model_ds()
    BASE_DIR = Path(__file__).resolve().parents[1]
    data = np.load(BASE_DIR / "train" / "weights.npz")
    linear_params = {k: data[k] for k in ["W1","b1","W2","b2","W3","b3","W4","b4"]}
    bn_params     = {k: data[k] for k in ["gamma1","beta1","gamma2","beta2","gamma3","beta3"]}
    bn_running    = {k: data[k] for k in ["running_mean1","running_var1",
                                        "running_mean2","running_var2",
                                        "running_mean3","running_var3"]}
    model = dense_model(linear_params, bn_params, bn_running)
    yhat = model.predict(x_test_dm)
    a = accuracy(yhat, y_test)
    cm = confusion_matrix(yhat, y_test)
    print(f"Accuracy: {a}")
    print(cm)
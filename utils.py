import matplotlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
from config import config
import csv

# set the matplotlib backend so figures can be saved in the background
matplotlib.use("Agg")


# resize an image to a certain width and keep ratio
def resize(image, width=800):
    (h, w) = image.shape[:2]
    ratio = width / float(w)
    height = int(h * ratio)
    ret_img = cv2.resize(image, (width, height))
    return ret_img, ratio


def plot_loss_accuracy(H):
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = config.EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower right")
    plt.savefig(config.MYVGG_PLOT_PATH)


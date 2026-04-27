from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import tifffile as tiff

AXES = "TZCYX"

def plot_train_validation_loss(train_loss, validation_loss, odirname):
    last_train_loss = round(train_loss[-1], 3)
    last_val_loss = round(validation_loss[-1], 3)

    plt.figure()
    plt.plot(train_loss, color = "blue")
    plt.plot(validation_loss, color = "orange")
    
    plt.annotate(
    f"Last train loss {last_train_loss}",
    xy=(0.98, 0.98),
    xycoords="axes fraction",
    ha="right",
    va="top",
    fontsize=5)

    plt.annotate(
    f"Last validation loss {round(last_val_loss, 3)}",
    xy=(0.98, 0.95),
    xycoords="axes fraction",
    ha="right",
    va="top",
    fontsize=5)

    plt.savefig(
        f"{odirname}/Train_Validation_loss.png",
        dpi=300,
        bbox_inches="tight",
    )

def plot_confusion_matrix(confusion_matrix, odirname):

    hm = sns.heatmap(data=confusion_matrix,
                annot=True,
                cmap="tab20b",
                fmt = ".0f",
                cbar=False)

    plt.xlabel("Predicted label")
    plt.ylabel("True label")


    plt.savefig(
        f"{odirname}/Confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )


def show_split_pairs(splits, n=6, seed=0): 

    for i in splits:
        xi = x[i][0]  # FIRST CHANNEL ONLY
        zi = z[i][0]  # FIRST CHANNEL ONLY

        fig, ax = plt.subplots(1, 3, figsize=(9, 3))

        ax[0].imshow(xi, cmap="gray")
        ax[0].set_title(f"{split} input")
        ax[0].axis("off")

        ax[1].imshow(zi, cmap="gray")
        ax[1].set_title("mask")
        ax[1].axis("off")

        # overlay
        ax[2].imshow(xi, cmap="gray")
        ax[2].imshow(zi, alpha=0.4)
        ax[2].set_title("overlay")
        ax[2].axis("off")

        plt.tight_layout()
        plt.show()


def model_validation_preview(dataloader, model, odirname, device):  #write single concat tiff instead


    final_concat = None

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_binary = torch.argmax(pred, dim = 1, keepdim=True)
            x.to(torch.uint16)
            x = x * 65535
            x0 = x.detach().cpu().numpy().astype(np.uint16)
            y0 = y.detach().cpu().numpy().astype(np.uint16)
            y0[y0 > 0] = 65535
            p0 = pred_binary.detach().cpu().numpy().astype(np.uint16)
            p0[p0 > 0] = 65535

            concat = np.concatenate((x0, y0, p0), axis = 1)

            if concat.ndim == 5:
                concat = np.transpose(concat, (0, 2, 1, 3, 4))

            if final_concat is None:
                final_concat = concat
            else:
                final_concat = np.concatenate((final_concat, concat), axis = 0)
        
        axes = AXES[-final_concat.ndim:]

        tiff.imwrite(
        f"{odirname}/model_test_performance.tif",
        final_concat,
        imagej=True,
        metadata={"mode": "color", "axes": axes, "unit": "micron", "spacing":0.5},
        resolution=(5.884696082960086, 5.884696082960086)
        )
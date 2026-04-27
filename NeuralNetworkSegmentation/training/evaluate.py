import torch
from sklearn.metrics import confusion_matrix
import numpy as np


def binary_one_hot_output_transform(pred_y, y):

    pred_y = torch.argmax(pred_y, dim = 1).long().flatten().cpu()

    y = y.flatten().cpu()

    return pred_y, y


def test_model(dataloader, model, loss_fn, out_channels, device):

    total_loss = 0

    final_con_mat = list()

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            pred = model(x)

            total_loss += loss_fn(pred, y).item()

            pred_y_flat, y_flat = binary_one_hot_output_transform(pred, y)
            con_mat = confusion_matrix(y_flat, pred_y_flat, labels=[i for i in range(out_channels)])
            final_con_mat.append(con_mat)
    

    final_con_mat = np.sum(final_con_mat, axis = 0)

    test_loss = total_loss / len(dataloader)
    print(f"Testing Loss: {test_loss:>7f}")

    print("\nConfusion Matrix\n", final_con_mat)

    return final_con_mat


def dice_score_coefficient(): # TODO
    pass
import numpy as np
import torch

from NeuralNetworkSegmentation.training.loops import train_one_epoch, validate_one_epoch

# def compute_class_weights_from_split(train_split, n_classes):  #conver the script to run using the array of train not the dataloader
#     class_counts = np.zeros(n_classes)

#     total_pixels = 0

#     for _, _, y in train_split:
        
#         y = (y > 0).astype(np.int64)
#         total_pixels += y.size
#         class_counts += np.bincount(
#             y.flatten(), minlength=n_classes
#         )
#         print(class_counts)

#     print("Classes distribution:", f"{[f'Class {idx}: {i}' for idx, i in enumerate(class_counts/total_pixels)]}\n")
#     weights = total_pixels / (n_classes * class_counts)
#     return weights


def compute_class_weights_from_dataloader(dataloader, num_classes, device):  #conver the script to run using the array of train not the dataloader
    class_counts = torch.zeros(num_classes, device=device)
    total_pixels = 0
    for _, y in dataloader:
        y = y.to(device)
        total_pixels += y.numel()
        class_counts += torch.bincount(
            y.flatten().type(torch.int),
            minlength=num_classes
        )

    print("Classes distribution:", f"{[f'Class {idx}: {i}' for idx, i in enumerate(class_counts/total_pixels)]}\n")
    weights = total_pixels / (num_classes * class_counts)
    return weights


def fit(num_epochs,
        optimizer,
        train_dataloader,
        validation_dataloader,
        model,
        loss_fn,
        device):

    train_losses = list()
    validation_losses = list()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")

        # if not epoch % 200 and epoch != 0: # CHANGE TO 100
        #     lr = optimizer.param_groups[0]['lr']
        #     optimizer.param_groups[0]['lr'] = lr / 10

        train_loss = train_one_epoch(train_dataloader, model, loss_fn, optimizer, device)
        train_losses.append(train_loss)
        
        validation_loss = validate_one_epoch(validation_dataloader, model, loss_fn, device)
        validation_losses.append(validation_loss)
    
    return train_losses, validation_losses

import torch

def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    total_loss = 0

    model.train()
    for batch_idx, batch in enumerate(dataloader):
        x, y = batch

        x, y = x.to(device), y.to(device)

        pred = model(x)
        
        loss = loss_fn(pred, y)

        if not torch.isfinite(loss):
            unique_targets = torch.unique(y).detach().cpu().tolist()
            raise ValueError(
                f"Non-finite loss at batch {batch_idx}. "
                f"Unique targets={unique_targets}, "
                f"pred min={pred.nan_to_num().min().item()}, pred max={pred.nan_to_num().max().item()}"
            )

        optimizer.zero_grad()
        loss.backward()

        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise ValueError(
                    f"Non-finite gradient found after backward at batch {batch_idx} "
                    f"for parameter '{name}'."
                )

        optimizer.step()        

        total_loss += loss.item()

    train_loss = total_loss / len(dataloader)
    print(f"Training Loss: {train_loss:>7f}")

    return train_loss # Average loss to plot # can maybe return pred ?? for like one batch


def validate_one_epoch(dataloader, model, loss_fn, device):

    total_loss = 0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            pred = model(x)

            total_loss += loss_fn(pred, y).item()

    valid_loss = total_loss / len(dataloader)
    print(f"Validating Loss: {valid_loss:>7f}")

    return valid_loss
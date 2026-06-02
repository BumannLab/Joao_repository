import numpy as np
import torch

from ZOrganized_scripts.NeuralNetworkSegmentationConfig.models.unet import is_model_2d
from ZOrganized_scripts.NeuralNetworkSegmentationConfig.metrics import compute_confusion_matrix

# TODO: PASS THIS TO UTILS OR LOSS FN
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


class ModelTrainer:
    """
    Args:
        model: UNet 3D model to be trained.
        optimizer: Optimizer used for training.
        lr_scheduler: Learning rate scheduler. Note that lr_scheduler.step() is invoked after every validation
            step (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion: Loss function.
        eval_criterion: Used to compute training/validation metric (such as Dice, IoU, AP or Rand score).
            Saving the best checkpoint is based on the result of this function on the validation set.
        loaders: Dictionary with 'train' and 'val' data loaders.
        checkpoint_dir: Directory for saving checkpoints and tensorboard logs.
        max_num_epochs: Maximum number of epochs.
        max_num_iterations: Maximum number of iterations.
        validate_after_iters: Validate after that many iterations. Default: 200.
        log_after_iters: Number of iterations before logging to tensorboard. Default: 100.
        validate_iters: Number of validation iterations. If None validate on the whole validation set. Default: None.
        num_iterations: Useful when loading the model from the checkpoint. Default: 1.
        num_epoch: Useful when loading the model from the checkpoint. Default: 0.
        eval_score_higher_is_better: If True higher eval scores are considered better. Default: True.
        tensorboard_formatter: Converts a given batch of input/output/target image to a series of images
            that can be displayed in tensorboard. Default: None.
        skip_train_validation: If True eval_criterion is not evaluated on the training set (used when
            evaluation is expensive). Default: False.
        resume: Path to the checkpoint to be resumed. Default: None.
        pre_trained: Path to the pre-trained model. Default: None.
        max_val_images: Maximum number of images to log during validation. Default: 100.
        device: Device to use for training (CPU, CUDA, MPS). Default: None.
    """
    
    def __init__(self,
        save_dir,
        save_model_name,
        model,
        optimizer,
        lr_scheduler,
        loss_criterion,
        eval_criterion,
        loaders,
        max_num_epochs,
        device,
    ):
        self.save_dir = save_dir
        self.save_model_name = save_model_name
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.loaders = loaders
        self.max_num_epochs = max_num_epochs
        self.device = device


    def train(self):

        epoch_loss = 0
        epoch_eval_score = 0
        
        self.model.train()

        for batch_idx, batch in enumerate(self.loaders["train"]):

            input, target = batch
            input, target = input.to(self.device), target.to(self.device)

            output, loss = self._forward_pass_(input, target)

            eval_score = self.eval_criterion(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            epoch_loss += loss.item()
            epoch_eval_score += eval_score

        epoch_loss = epoch_loss / len(self.loaders["train"])
        epoch_eval_score = epoch_eval_score / len(self.loaders["train"])
        
        print(f"Training stats. Loss: {epoch_loss:>9f}. Evaluation score: {epoch_eval_score:>9f}")

        return epoch_loss, epoch_eval_score
    

    def validation(self):

        epoch_loss = 0
        epoch_eval_score = 0
        
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.loaders["val"]):

                input, target = batch
                input, target = input.to(self.device), target.to(self.device)

                output, loss = self._forward_pass_(input, target)
                
                eval_score = self.eval_criterion(output, target)

                epoch_loss += loss.item()
                epoch_eval_score += eval_score

            epoch_loss = epoch_loss / len(self.loaders["train"])
            epoch_eval_score = epoch_eval_score / len(self.loaders["train"])

            print(f"Validation stats. Loss: {epoch_loss:>9f}. Evaluation score: {epoch_eval_score:>9f}")

            return epoch_loss, epoch_eval_score


    def test(self):

        epoch_loss = 0
        epoch_eval_score = 0

        final_confusion_matrix = []

        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.loaders["val"]):

                input, target = batch
                input, target = input.to(self.device), target.to(self.device)

                output, loss = self._forward_pass_(input, target)
                
                eval_score = self.eval_criterion(output, target)
                
                confusion_matrix = compute_confusion_matrix(output, target)
                final_confusion_matrix.append(confusion_matrix)

                epoch_loss += loss.item()
                epoch_eval_score += eval_score

            epoch_loss = epoch_loss / len(self.loaders["train"])
            epoch_eval_score = epoch_eval_score / len(self.loaders["train"])

            print(f"Test stats. Loss: {epoch_loss:>9f}. Evaluation score: {epoch_eval_score:>9f}")

            final_confusion_matrix = np.sum(final_confusion_matrix, axis = 0)
        
            return epoch_loss, epoch_eval_score, final_confusion_matrix 


    def fit(self):

        train_losses = []
        train_eval_scores = []

        validation_losses = []
        validation_eval_scores = []

        for epoch in range(self.max_num_epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")

            train_loss, train_eval = self.train()

            train_losses.append(train_loss)
            train_eval_scores.append(train_eval)

            val_loss, val_eval = self.validation()

            validation_losses.append(val_loss)
            validation_eval_scores.append(val_eval)

        test_loss, test_eval_score, confusion_matrix = self.test()

        print(confusion_matrix)


    def _forward_pass_(self, input, target): # TODO: Transpose output into correct format??
        """
        Handles 2d and 3d inputs if the model is 2d it will remove the singleton z-dimension
        """
        if is_model_2d(self.model):

            # remove the singleton z-dimension from the input
            input = torch.squeeze(input, dim=-3)
            
            # forward pass
            output, logits = self.model(input)
            
            # add the singleton z-dimension to the output
            output = torch.unsqueeze(output, dim=-3)
            print(output.shape)
            logits = torch.unsqueeze(logits, dim=-3)
        
        else:

            # forward pass
            output, logits = self.model(input)

        # always compute the loss using logits
        loss = self.loss_criterion(logits, target)

        # return probabilities and loss
        return output, loss

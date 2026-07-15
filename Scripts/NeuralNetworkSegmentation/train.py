import os
from datetime import datetime

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from Scripts.NeuralNetworkSegmentation.models.unet import is_model_2d
from Scripts.NeuralNetworkSegmentation.metrics import compute_confusion_matrix
from Scripts.NeuralNetworkSegmentation.utils import save_checkpoint, load_checkpoint, get_logger, convert_to_numpy
from Scripts.NeuralNetworkSegmentation.visualization import TensorboardFormatter

logger = get_logger("ModelTrainer")

class ModelTrainer:
    """
    Trains and validates a segmentation model, handling the epoch loop, checkpointing,
    early stopping, learning rate scheduling, and TensorBoard logging.

    Runs one training and one validation pass per epoch. Scalar metrics (loss, eval score,
    learning rate) are logged to TensorBoard every epoch. Validation image results are logged
    only every `log_after_epochs` epochs, since validation itself always runs every epoch but
    writing image summaries that often would be wasteful.

    Args:
        model: UNet 3D model to be trained.
        optimizer: Optimizer used for training.
        lr_scheduler: Learning rate scheduler. Invoked after every epoch
        loss_criterion: Loss function.
        eval_criterion: Used to compute training/validation metric (such as Dice, IoU, AP or Rand score).
            Saving the best checkpoint is based on the result of this function on the validation set.
        loaders: Dictionary with 'train' and 'val' data loaders.
        checkpoint_dir: Directory for saving checkpoints and tensorboard logs.
        max_num_epochs: Maximum number of epochs.
        early_stopping: Default: True
        max_epochs_no_improvement: Default: 100
        num_epochs: Useful when loading the model from the checkpoint. Default: 0.
        eval_score_higher_is_better: If True higher eval scores are considered better. Default: True.
        log_after_epochs: Number of epochs between validation image logging to tensorboard. Validation
            itself still runs every epoch; only the image summaries are skipped in between. Default: 100.
        max_val_images: Maximum number of images to log during validation. Default: 100. 
        skip_train_validation: If True eval_criterion is not evaluated on the training set (used when
            evaluation is expensive). Default: False.
        resume: Path to the checkpoint to be resumed. Default: None.
        pre_trained: Path to the pre-trained model. Default: None.
        device: Device to use for training (CPU, CUDA, MPS). Default: None.
    """
    
    def __init__(self,
        model,
        optimizer,
        lr_scheduler,
        loss_criterion,
        eval_criterion,
        loaders,
        checkpoint_dir = "", # Maybe remove the "" and put None or make it essential to have
        channels_to_load = None,
        max_num_epochs = 1000,
        num_epochs = 0,
        early_stopping = True,
        max_patience_epochs = 100,
        eval_score_higher_is_better = True,
        skip_train_validation = False, # not being used
        resume = False,
        pre_trained = False,
        log_after_epochs = 50,
        max_val_images = 100,
        device = None
    ):
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.loaders = loaders
        self.channels_to_load = channels_to_load

        self.checkpoint_dir = checkpoint_dir
        
        self.max_num_epochs = max_num_epochs
        self.num_epochs = num_epochs

        self.log_after_epochs = log_after_epochs
        self.max_val_images = max_val_images

        self.skip_train_validation = skip_train_validation

        self.eval_score_higher_is_better = eval_score_higher_is_better
        
        assert device, "Device must be specified"
        self.device = device

        # initialize the best_eval_score
        if eval_score_higher_is_better:
            self.best_eval_score = float("-inf")
        else:
            self.best_eval_score = float("+inf")

        self.early_stopping = early_stopping
        self.max_patience_epochs = max_patience_epochs # Number of epochs with no improvement
        self.num_patience_epochs = 0

        self.writer = SummaryWriter(
            log_dir=os.path.join(checkpoint_dir, "logs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        )

        self.tensorboard_formatter = TensorboardFormatter()

        if resume is not None:
            
            state = load_checkpoint(resume, self.model, self.optimizer)

            self.best_eval_score = state["best_eval_score"]
            self.num_epochs = state["num_epochs"]
            self.checkpoint_dir = os.path.split(resume)[0]
        
        elif pre_trained is not None:

            load_checkpoint(pre_trained, self.model, None)
            
            if not self.checkpoint_dir:
                self.checkpoint_dir = os.path.split(pre_trained)[0]


    def train(self):

        epoch_loss = 0

        self.eval_criterion.reset()
        self.model.train()

        for batch_idx, batch in enumerate(self.loaders["train"]):

            input, target = batch # BC[D]HW , B1[D]HW ; 1 can be equal to C if OneHot Augmentation is used
            input, target = input.to(self.device), target.to(self.device)

            output, loss = self._forward_pass_(input, target)

            self.eval_criterion(output, target)

            self.optimizer.zero_grad()
            loss.backward()

            # NOTE: Optional log Gradients
            # if batch_idx == 0 and self.num_epochs % self.log_after_epochs == 0:
            #     self._log_params()

            self.optimizer.step()
            
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(self.loaders["train"])
        epoch_eval_score = self.eval_criterion.aggregate(reduction="mean").item()
        
        logger.info(f"Training stats. Loss: {epoch_loss:>9f}. Evaluation score: {epoch_eval_score:>9f}")

        return epoch_loss, epoch_eval_score
    

    def validation(self):
        """
        Runs one full validation pass. Validation runs every epoch, but validation image
        results are only logged to tensorboard every `log_after_epochs` epochs so that we're
        not writing image summaries on every single epoch, which would be a waste.
        """

        epoch_loss = 0

        self.eval_criterion.reset()
        self.model.eval()

        log_images = self.num_epochs % self.log_after_epochs == 0

        with torch.no_grad():

            if log_images:
                rs = np.random.RandomState(42)
                if len(self.loaders["val"]) <= self.max_val_images:
                    indices = set(range(len(self.loaders["val"])))
                else:
                    indices = set(rs.choice(len(self.loaders["val"]), size=self.max_val_images, replace=False))
                preview_idx = 0

            for batch_idx, batch in enumerate(self.loaders["val"]):

                input, target = batch
                input, target = input.to(self.device), target.to(self.device)

                output, loss = self._forward_pass_(input, target)

                self.eval_criterion(output, target)

                epoch_loss += loss.item()

                if log_images and batch_idx in indices:
                    preview_idx = self._log_images(input, target, output, preview_idx, phase="val")

            epoch_loss = epoch_loss / len(self.loaders["val"])
            epoch_eval_score = self.eval_criterion.aggregate(reduction="mean").item()

            self.writer.add_scalar('Loss/val', epoch_loss, self.num_epochs)
            self.writer.add_scalar('Eval/val', epoch_eval_score, self.num_epochs)

            logger.info(f"Validation stats. Loss: {epoch_loss:>9f}. Evaluation score: {epoch_eval_score:>9f}")

            return epoch_loss, epoch_eval_score


    # def test(self):

    #     epoch_loss = 0

    #     final_confusion_matrix = []

    #     self.eval_criterion.reset()
    #     self.model.eval()

    #     preview_idx = 0

    #     with torch.no_grad():
    #         for batch_idx, batch in enumerate(self.loaders["test"]):

    #             input, target = batch
    #             input, target = input.to(self.device), target.to(self.device)

    #             output, loss = self._forward_pass_(input, target)

    #             self.eval_criterion(output, target)
                
    #             confusion_matrix = compute_confusion_matrix(output, target)
    #             final_confusion_matrix.append(confusion_matrix)

    #             epoch_loss += loss.item()

    #             model_test_preview(self.log_writer, input, target, output, i=preview_idx)
    #             preview_idx += output.shape[0]

    #         epoch_loss = epoch_loss / len(self.loaders["test"])
    #         epoch_eval_score = self.eval_criterion.aggregate(reduction="mean").item()

    #         print(f"Test stats. Loss: {epoch_loss:>9f}. Evaluation score: {epoch_eval_score:>9f}")

    #         final_confusion_matrix = np.sum(final_confusion_matrix, axis = 0)
        
    #         return epoch_loss, epoch_eval_score, final_confusion_matrix 


    def fit(self):

        for _ in range(self.num_epochs, self.max_num_epochs):
            print(f"Epoch {self.num_epochs+1}\n-------------------------------")

            train_loss, train_eval = self.train()
            
            self.writer.add_scalar('Loss/train', train_loss, self.num_epochs)
            self.writer.add_scalar('Eval/train', train_eval, self.num_epochs)
           
            val_loss, val_eval = self.validation()

            self.writer.add_scalar('Loss/val', val_loss, self.num_epochs)
            self.writer.add_scalar('Eval/val', val_eval, self.num_epochs)

            self._log_lr()
                
            # adjust learning rate if necessary
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_eval)
            elif self.scheduler is not None:
                self.scheduler.step()

            is_best = self._is_best_eval_score(val_eval)

            if is_best:
                self.num_patience_epochs = 0
            else:
                self.num_patience_epochs += 1

            self.num_epochs += 1

            self._save_checkpoint(is_best)

            if self.should_stop(): # Stop if lr is low or no improvement
                break 

        self.writer.close()

    
    def _is_best_eval_score(self, eval_score: float) -> bool:
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            logger.info(f"Saving new best evaluation metric: {eval_score}")
            self.best_eval_score = eval_score

        return is_best
    
    def _forward_pass_(self, input, target):
        """
        Handles 2d and 3d inputs if the model is 2d it will remove the singleton z-dimension
        """

        input = input[:, self.channels_to_load]

        if is_model_2d(self.model):

            # remove the singleton z-dimension from the input
            input = torch.squeeze(input, dim=-3)
            
            # forward pass
            output, logits = self.model(input, return_logits = True)
            
            # add the singleton z-dimension to the output
            output = torch.unsqueeze(output, dim=-3)
            logits = torch.unsqueeze(logits, dim=-3)
        
        else:

            # forward pass
            output, logits = self.model(input, return_logits = True)

        # always compute the loss using logits
        loss = self.loss_criterion(logits, target)

        # return probabilities and loss
        return output, loss
        
    
    def should_stop(self):
        """Check if training should be terminated.

        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case).

        Returns:
            True if training should stop, False otherwise.
        """
        if self.early_stopping and self.num_patience_epochs >= self.max_patience_epochs:
            logger.info(f"No improvement in evaluation score for {self.num_patience_epochs} epochs.")
            return True
        
        min_lr = 1e-6
        lr = self.optimizer.param_groups[0]["lr"]
        if lr < min_lr:
            logger.info(f"Learning rate below the minimum {min_lr}.")
            return True

        return False

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("learning_rate", lr, self.num_epochs)

    def _log_params(self):
        logger.info("Logging model parameters and gradients")
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_epochs)
            self.writer.add_histogram(name + "/grad", value.grad.data.cpu().numpy(), self.num_epochs)
    
    def _log_images(self, input, target, prediction, i: int, phase: str) -> int:
        """
        Logs one preview image (input/target/prediction) per sample in the batch via
        self.tensorboard_formatter. Returns the next free preview index.
        """
        for c_input, c_target, c_pred in zip(input, target, prediction):

            if c_pred.shape[0] > 1:
                c_pred = torch.argmax(c_pred, dim=0, keepdim=True)
            else:
                c_pred = (c_pred > 0.5).float()

            c_input, c_target, c_pred = convert_to_numpy(c_input, c_target, c_pred)

            input_img = self.tensorboard_formatter(c_input)
            target_img = self.tensorboard_formatter(c_target)
            pred_img = self.tensorboard_formatter(c_pred)

            self.writer.add_image(f"ModelPreview/{phase}/{i}Input", input_img, self.num_epochs, dataformats='HW' if input_img.ndim == 2 else 'HWC')
            self.writer.add_image(f"ModelPreview/{phase}/{i}Target", target_img, self.num_epochs, dataformats='HW' if target_img.ndim == 2 else 'HWC')
            self.writer.add_image(f"ModelPreview/{phase}/{i}Pred", pred_img, self.num_epochs, dataformats='HW' if pred_img.ndim == 2 else 'HWC')

            i += 1

        return i


    def _save_checkpoint(self, is_best: bool):

        state_dict = self.model.state_dict()

        save_checkpoint(
            {
                "num_epochs": self.num_epochs + 1,
                "model_state_dict": state_dict,
                "best_eval_score": self.best_eval_score,
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            is_best,
            checkpoint_dir=self.checkpoint_dir,
        )
        
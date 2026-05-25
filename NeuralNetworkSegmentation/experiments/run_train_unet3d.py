from pathlib import Path
from datetime import datetime
import json
import os

import torch
from torch.utils.data import DataLoader
from monai.losses import DiceFocalLoss

from NeuralNetworkSegmentation.data.io import load_images
from NeuralNetworkSegmentation.data.splits import ordered_split
from NeuralNetworkSegmentation.data.transforms import (
    AdjustBrightness,
    CenterCrop,
    ComposePair,
    DepthPadding,
    GaussianBlur,
    RandomAffine,
    RandomHorizontalFlip,
    RandomVerticalFlip
)
from NeuralNetworkSegmentation.data.datasets import BacteriaMaskPair
from NeuralNetworkSegmentation.models.unet import UNet
from NeuralNetworkSegmentation.training.losses import FocalLoss, DiceWeightedCrossEntropyLoss, SoftDiceLoss
from NeuralNetworkSegmentation.training.train import (
    compute_class_weights_from_dataloader,
    fit
)
from NeuralNetworkSegmentation.training.evaluate import test_model
from NeuralNetworkSegmentation.utils.visualization import (
    plot_train_validation_loss,
    plot_confusion_matrix,
    model_validation_preview
)
from NeuralNetworkSegmentation.utils.seed import seed_everything

torch.cuda.empty_cache()

######################
# IMAGE & MASKS PATH #
######################
INPUT_DIR = "Data/DATASET_COMPARISON/3D_comparison/Dataset_threshold_800/inputs"  # Directory containing the training input volumes.
CHANNELS_TO_LOAD = [0]  # Input channels to keep from each volume, using 0-based indexing.
MASK_DIR = "Data/DATASET_COMPARISON/3D_comparison/Dataset_threshold_800/masks"  # Directory containing the segmentation mask volumes.


##############
# DATA SPLIT #
##############
SPLIT_TRAIN = 0.6  # Fraction of samples assigned to the training split.
SPLIT_VALIDATION = 0.2  # Fraction of samples assigned to the validation split.
SPLIT_TEST = 0.2  # Fraction of samples assigned to the test split.


###################
# DATA GENERATION #
###################
GENERATE_DATA = True  # Whether to generate augmented variants from each training sample.
NUM_IMAGES = 10  # Number of augmented samples to generate per original training sample.


#####################
# DATA AUGMENTATION #
#####################
BRIGHTNESS_FACTOR = (0.5, 1.0)  # Brightness jitter range applied during augmentation.
SAME_CHANNEL_BRIGHTNESS = False  # Whether all channels share the same sampled brightness factor.
KERNEL_SIZE = 3  # Gaussian blur kernel size used during augmentation.
SIGMA = (0.1, 1.5)  # Range of Gaussian blur sigma values.
PROBABILITY_BLUR = 0.4  # Probability of applying Gaussian blur to a sample.
HORIZONTAL_FLIP_PROB = 0.5
VERTICAL_FLIP_PROB = 0.5
DEGREES = 180  # Maximum rotation angle in degrees for random affine augmentation.
TRANSLATE = (0.2, 0.2) #(0.09, 0.09)  # Maximum horizontal and vertical translation as image fractions.
SCALE = (0.85, 1.15)  # Min and max scale factors for random affine augmentation.
SHEAR = (-8, 8)  # Min and max shear values for random affine augmentation.
SIZE = 128  # Final cropped spatial size used for training patches.
Z_PADDING = (2, 1)  # Depth padding added before/after the z axis.


#################
# LOSS FUNCTION #
#################
LOSS_FUNCTION = "None"  # Loss function selection key for this experiment.


#############
# OPTIMIZER #
#############
OPTIMIZER = "ADAM"  # Optimizer selection key for this experiment.
LEARNING_RATE = 0.0001  # Initial learning rate passed to the optimizer.
WEIGHT_DECAY = 0


##############
# PARAMETERS #
##############
BATCH_SIZE = 32  # Number of samples processed per optimization step.
NUM_EPOCHS = 150  # Total number of full passes through the training dataset.


#########
# MODEL #
#########
MODEL = "None" # 
INPUT_CHANNELS = len(CHANNELS_TO_LOAD)  # Number of input channels expected by the network.
FEATURE_CHANNELS = 32  # Base number of feature channels in the first UNet block.
OUTPUT_CHANNELS = 2  # Number of output classes predicted by the network.
NUM_BLOCKS = 3  # Number of encoder/decoder stages in the UNet.
USE_BATCHNORM = True  # Whether to use batch normalization inside convolution blocks.
MODEL_ODIRNAME = "./mTB_segmentation/2Dvs3D/recreating_attempt1_3D/model"  # Output path used to save the trained model.
ALLOW_MODEL_OVERWRITE = False  # Set to True only when you intentionally want to replace an existing .pt file.

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"  # Device used for training and inference.

SEED = 2168897548 #int.from_bytes(os.urandom(4), 'big')  # Random seed used to make data order, augmentation, and training reproducible.
print(SEED)
# For random seeds use int.from_bytes(os.urandom(4), 'big')


def save_run_globals():
    run_time = datetime.now().astimezone()
    run_config = {
        "run_time": run_time.isoformat(),
        "script": str(Path(__file__).resolve()),
        "globals": {
            name: value
            for name, value in globals().items()
            if name.isupper()
        },
    }

    odirname = Path(MODEL_ODIRNAME).parent / "run_configs"
    odirname.mkdir(parents=True, exist_ok=True)

    timestamp = run_time.strftime("%Y%m%d_%H%M%S")
    output_path = odirname / f"run_config_{timestamp}_seed_{SEED}.json"

    with open(output_path, "w") as outfile:
        json.dump(run_config, outfile, indent=2, default=str)

    print(f"Saved run config: {output_path}")
    return output_path


def get_model_output_path(model_odirname):
    return Path(f"{model_odirname}.pt")


def ensure_model_output_available(model_odirname, allow_overwrite=False):
    output_path = get_model_output_path(model_odirname)

    if output_path.exists() and not allow_overwrite:
        raise FileExistsError(
            f"Model already exists: {output_path}. "
            "Change MODEL_ODIRNAME or set ALLOW_MODEL_OVERWRITE = True intentionally."
        )

    return output_path


def save_model_state(model, model_odirname, allow_overwrite=False):
    output_path = ensure_model_output_available(model_odirname, allow_overwrite)

    torch.save(model.state_dict(), output_path)
    print(f"Saved model: {output_path}")
    return output_path

def main(input_dir,
         channels_to_load,
         mask_dir,
         split_train,
         split_val,
         split_test,
         generate_data,
         num_images,
         brightness_factor,
         same_channel_brightness,
         kernel_size,
         sigma,
         probability_blur,
         horizontal_flip_prob,
         vertical_flip_prob,
         degrees,
         translate,
         scale,
         shear,
         size,
         z_padding,
         loss_function, # not used for now
         optimizer_name, # not used for now
         learning_rate,
         weight_decay,
         batch_size,
         num_epochs,
         model_name, # not used for now
         input_channels,
         feature_channels,
         output_channels,
         num_blocks,
         use_batchnorm,
         model_odirname,
         device,
         seed):
    
    save_run_globals()
    ensure_model_output_available(model_odirname, ALLOW_MODEL_OVERWRITE)
    seed_everything(seed)
    
    grouped_input_mask_pairs = load_images(input_dir, mask_dir, channels_to_load)

    splits = ordered_split(grouped_input_mask_pairs, split_train, split_val, split_test)

    transform_3d = ComposePair(
        (AdjustBrightness(brightness_factor, same_channel_brightness),
        RandomAffine(degrees, translate, scale, shear),
        GaussianBlur(kernel_size, sigma, probability_blur),
        DepthPadding(z_padding),
        CenterCrop(size),
        RandomHorizontalFlip(horizontal_flip_prob),
        RandomVerticalFlip(vertical_flip_prob),)

    )

    train_dataset = BacteriaMaskPair(splits["train"], transform=transform_3d, generate_data = generate_data, num_images = num_images)
    validation_dataset = BacteriaMaskPair(splits["val"], transform=ComposePair((CenterCrop(size), DepthPadding(z_padding))))
    test_dataset  = BacteriaMaskPair(splits["test"], transform=ComposePair((CenterCrop(size), DepthPadding(z_padding))))
     
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Train Dataset: ", len(train_dataloader.dataset), " Images")
    print("Validation Dataset: ", len(validation_dataloader.dataset), " Images")
    print("Test Dataset: ", len(test_dataloader.dataset), " Images")
    print()

    model = UNet(input_channels, output_channels, feature_channels, num_blocks = num_blocks, use_batchnorm = use_batchnorm, is_3d=True).to(device)

    class_weights = compute_class_weights_from_dataloader(train_dataloader, output_channels, device)
    loss_fn = DiceFocalLoss(
            to_onehot_y=True,
            softmax=True,
            include_background=True,
            smooth_dr=1.0,
            smooth_nr=1.0,
            lambda_dice=0.6,
            gamma=1.0,
            lambda_focal=0.4,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses, validation_losses = fit(num_epochs, optimizer, train_dataloader, validation_dataloader, model, loss_fn, device)
    confusion_matrix = test_model(test_dataloader, model, loss_fn, output_channels, device)
    
    odirname = Path(model_odirname).parent
    odirname.mkdir(parents=True, exist_ok=True)

    save_model_state(model, model_odirname, ALLOW_MODEL_OVERWRITE)

    model_validation_preview(test_dataloader, model, odirname, device)
    
    plot_train_validation_loss(train_losses, validation_losses, odirname)
    plot_confusion_matrix(confusion_matrix, odirname)

    
if __name__ == "__main__":
    main(
        INPUT_DIR,
        CHANNELS_TO_LOAD,
        MASK_DIR,
        SPLIT_TRAIN,
        SPLIT_VALIDATION,
        SPLIT_TEST,
        GENERATE_DATA,
        NUM_IMAGES,
        BRIGHTNESS_FACTOR,
        SAME_CHANNEL_BRIGHTNESS,
        KERNEL_SIZE,
        SIGMA,
        PROBABILITY_BLUR,
        HORIZONTAL_FLIP_PROB,
        VERTICAL_FLIP_PROB,
        DEGREES,
        TRANSLATE,
        SCALE,
        SHEAR,
        SIZE,
        Z_PADDING,
        LOSS_FUNCTION,
        OPTIMIZER,
        LEARNING_RATE,
        WEIGHT_DECAY,
        BATCH_SIZE,
        NUM_EPOCHS,
        MODEL,
        INPUT_CHANNELS,
        FEATURE_CHANNELS,
        OUTPUT_CHANNELS,
        NUM_BLOCKS,
        USE_BATCHNORM,
        MODEL_ODIRNAME,
        DEVICE,
        SEED
    )





































# # TODO: Add a way to handle if class is just 2 (background and label) in function merge_mask_label_images

# ######################
# # IMAGE & MASKS PATH #
# ######################
# IMAGE_DIR =   "/data/joao/Materials/Joao_repository/NN_Dataset/DATASET_COMPARISON/3D_comparison/Dataset_threshold_800/inputs"
# MASK_IMG_DIR =  "/data/joao/Materials/Joao_repository/NN_Dataset/DATASET_COMPARISON/3D_comparison/Dataset_threshold_800/masks"


# ##############
# # DATA SPLIT #
# ##############
# SPLIT_TRAIN = 0.6
# SPLIT_VALIDATION = 0.2
# SPLIT_TEST = 0.2


# #####################
# # DATA AUGMENTATION #
# #####################
# GENERATE_DATA = False
# NUM_IMAGES = 20
# TRANSFORM_FN = transform3d


# #################
# # LOSS FUNCTION #
# #################
# LOSS_FUNCTION = "None" # TODO: This does not work for now, change on the main code # Options: SoftDice, CrossEntropy, DICE+CE, IoU (Intersection over Union), FocalLoss


# #############
# # OPTIMIZER #
# #############
# OPTIMIZER = "ADAM" # TODO: This does not work for now, change on the main code # Options: SGD, ADAM
# LEARNING_RATE = 0.0001


# ##############
# # PARAMETERS #
# ##############
# BATCH_SIZE = 16
# NUM_EPOCHS = 300


# #########
# # MODEL #
# #########
# INPUT_CHANNELS = 1
# FEATURE_CHANNELS = 32
# OUTPUT_CHANNELS = 2
# NUM_BLOCKS = 3
# USE_BATCHNORM = True
# MODEL_ODIRNAME = "/data/joao/Materials/Joao_repository/mTB_segmentation/Threshold800_3D/model"


# DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# ########
# # SEED #
# ########
# SEED = int.from_bytes(os.urandom(4), 'big')


# def seed_everything(seed): # DONE
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)


# #########
# # PLOTS #
# #########


# def plot_train_validation_loss(train_loss, validation_loss, odirname):
#     last_train_loss = round(train_loss[-1], 3)
#     last_val_loss = round(validation_loss[-1], 3)

#     plt.figure()
#     plt.plot(train_loss, color = "blue")
#     plt.plot(validation_loss, color = "orange")
    
#     plt.annotate(
#     f"Last train loss {last_train_loss}",
#     xy=(0.98, 0.98),
#     xycoords="axes fraction",
#     ha="right",
#     va="top",
#     fontsize=5)

#     plt.annotate(
#     f"Last validation loss {round(last_val_loss, 3)}",
#     xy=(0.98, 0.95),
#     xycoords="axes fraction",
#     ha="right",
#     va="top",
#     fontsize=5)

#     plt.savefig(
#         f"{odirname}/Train_Validation_loss.png",
#         dpi=300,
#         bbox_inches="tight",
#     )


# def plot_confusion_matrix(confusion_matrix, odirname):

#     hm = sns.heatmap(data=confusion_matrix,
#                 annot=True,
#                 cmap="tab20b",
#                 fmt = ".0f",
#                 cbar=False)

#     plt.xlabel("Predicted label")
#     plt.ylabel("True label")


#     plt.savefig(
#         f"{odirname}/Confusion_matrix.png",
#         dpi=300,
#         bbox_inches="tight",
#     )


# class BacteriaMaskPair(Dataset): # # DONE
#     def __init__(self, input_images: np.ndarray, label_images: np.ndarray, transform=None, generate_data=False, num_images=None):
#         self.input_images = input_images
#         self.label_images = label_images
#         self.transform = transform
#         self.generate_data = generate_data
#         self.num_images = num_images

#     def __len__(self):
#         if self.generate_data:
#             return len(self.input_images) * self.num_images
        
#         return len(self.input_images)
    
#     def __getitem__(self, idx: int):
#         if self.generate_data:
#             idx = idx // self.num_images
        
#         input_image = self.input_images[idx]
#         label_image = self.label_images[idx]

#         input_tensor = torch.from_numpy(input_image / 65535).to(torch.float32)
#         label_tensor = torch.from_numpy(label_image / 255).to(torch.long)

#         if self.transform:
#             input_tensor, label_tensor = self.transform(input_tensor, label_tensor)

#         if all([input_tensor.ndim == 4, label_tensor.ndim == 4]):
#             input_tensor = input_tensor.permute(1, 0, 2, 3) 
#             label_tensor = label_tensor.permute(1, 0, 2, 3) 

#         return input_tensor, label_tensor


# def load_images(folder: Path, is_mask = False):
#     """
#     Input: Path of the folder containing the images for training (Either
#     to load bacteria or mask of the bacteria images)
#     Output: Numpy array containing all the images
#     Process: TODO
#     """
#     images_path = get_images_path(folder, ".tif", ".tiff")

#     images = defaultdict(list)
#     for path in images_path:

#         image = tiff.imread(path)

#         if image.ndim == 3 and image.shape[-3] == 4: # for 2d data selects the sybrgold channel and creates a new axis because the indexing will make it HxW and not 1xHxW
#             image = image[0, ...]
#             image = image[np.newaxis, ...]

#         if image.ndim == 4 and image.shape[-3] == 4: # for 2d data selects the sybrgold channel and creates a new axis
#             image = image[:, 0, ...]
#             image = image[:, np.newaxis, ...]

#         if is_mask:
#             image = np.expand_dims(image, axis=-3)

#         if "negative" in path.name.lower():
#             images[0].append(image)

#         elif "positive" in path.name.lower():
#             images[1].append(image)

#         else:
#             print(f"No label in {path}: either negative or positive, found none")
    
#     images = {type: np.array(arr) for type, arr in images.items()}

#     return images


# def ordered_split(
#     input_dict: dict[int, np.ndarray],
#     mask_dict: dict[int, np.ndarray],
#     train_frac: float,
#     val_frac: float,
#     test_frac: float,
#     neg_ratio: float,
#     pos_ratio: float,
# ):
#     assert abs(train_frac + val_frac + test_frac - 1) < 1e-8
#     assert abs(neg_ratio + pos_ratio - 1) < 1e-8
#     assert len(input_dict[0]) == len(mask_dict[0]), "Class 0 input/mask mismatch"
#     assert len(input_dict[1]) == len(mask_dict[1]), "Class 1 input/mask mismatch"

#     # ---- Shuffle within each class ----
#     p0 = np.random.permutation(len(input_dict[0]))
#     p1 = np.random.permutation(len(input_dict[1]))

#     in0, mk0 = input_dict[0][p0], mask_dict[0][p0]
#     in1, mk1 = input_dict[1][p1], mask_dict[1][p1]

#     n_neg = len(in0)
#     n_pos = len(in1)

#     # ---- Split each class independently (FIXED) ----
#     tr_neg = int(round(n_neg * train_frac))
#     va_neg = int(round(n_neg * val_frac))
#     te_neg = n_neg - tr_neg - va_neg

#     tr_pos = int(round(n_pos * train_frac))
#     va_pos = int(round(n_pos * val_frac))
#     te_pos = n_pos - tr_pos - va_pos

#     neg_ptr = 0
#     pos_ptr = 0

#     def take_split(n_neg_take, n_pos_take):
#         nonlocal neg_ptr, pos_ptr

#         input_neg = in0[neg_ptr:neg_ptr + n_neg_take]
#         mask_neg  = mk0[neg_ptr:neg_ptr + n_neg_take]

#         input_pos = in1[pos_ptr:pos_ptr + n_pos_take]
#         mask_pos  = mk1[pos_ptr:pos_ptr + n_pos_take]

#         neg_ptr += n_neg_take
#         pos_ptr += n_pos_take

#         inp = np.concatenate([input_neg, input_pos], axis=0)
#         msk = np.concatenate([mask_neg,  mask_pos], axis=0)

#         return {"input": inp, "mask": msk}

#     splits = {
#         "train": take_split(tr_neg, tr_pos),
#         "val":   take_split(va_neg, va_pos),
#         "test":  take_split(te_neg, te_pos),
#     }

#     return splits


# def compute_class_weights_from_dataloader(dataloader, num_classes):  #conver the script to run using the array of train not the dataloader
#     class_counts = torch.zeros(num_classes, device=DEVICE)
#     total_pixels = 0
#     for _, y in dataloader:
#         y = y.to(DEVICE)
#         total_pixels += y.numel()
#         class_counts += torch.bincount(
#             y.flatten().type(torch.int),
#             minlength=num_classes
#         )

#     print("Classes distribution:", f"{[f'Class {idx}: {i}' for idx, i in enumerate(class_counts/total_pixels)]}\n")
#     weights = total_pixels / (num_classes * class_counts)
#     return weights


# def train_model(dataloader, model, loss_fn, optimizer):
#     total_loss = 0

#     model.train()
#     for batch_idx, batch in enumerate(dataloader):
#         x, y = batch

#         x, y = x.to(DEVICE), y.to(DEVICE)

#         if not torch.isfinite(x).all():
#             raise ValueError(
#                 f"Non-finite values found in input batch {batch_idx}. "
#                 f"x min={x.min().item()}, x max={x.max().item()}"
#             )

#         if not torch.isfinite(y).all():
#             raise ValueError(f"Non-finite values found in target batch {batch_idx}.")

#         pred = model(x)

#         if not torch.isfinite(pred).all():
#             raise ValueError(
#                 f"Non-finite values found in model output at batch {batch_idx}. "
#                 f"pred min={pred.nan_to_num().min().item()}, pred max={pred.nan_to_num().max().item()}"
#             )
        
#         loss = loss_fn(pred, y)

#         if not torch.isfinite(loss):
#             unique_targets = torch.unique(y).detach().cpu().tolist()
#             raise ValueError(
#                 f"Non-finite loss at batch {batch_idx}. "
#                 f"Unique targets={unique_targets}, "
#                 f"pred min={pred.nan_to_num().min().item()}, pred max={pred.nan_to_num().max().item()}"
#             )

#         optimizer.zero_grad()
#         loss.backward()

#         for name, parameter in model.named_parameters():
#             if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
#                 raise ValueError(
#                     f"Non-finite gradient found after backward at batch {batch_idx} "
#                     f"for parameter '{name}'."
#                 )

#         optimizer.step()        

#         total_loss += loss.item()

#     train_loss = total_loss / len(dataloader)
#     print(f"Training Loss: {train_loss:>7f}")

#     return train_loss # Average loss to plot # can maybe return pred ?? for like one batch


# def validate_model(dataloader, model, loss_fn):

#     total_loss = 0

#     model.eval()
#     with torch.no_grad():
#         for batch in dataloader:
#             x, y = batch
#             x, y = x.to(DEVICE), y.to(DEVICE)
#             pred = model(x)

#             total_loss += loss_fn(pred, y).item()

#     valid_loss = total_loss / len(dataloader)
#     print(f"Validating Loss: {valid_loss:>7f}")

#     return valid_loss


# def test_model(dataloader, model, loss_fn, out_channels):

#     total_loss = 0

#     final_con_mat = list()

#     model.eval()
#     with torch.no_grad():
#         for batch in dataloader:
#             x, y = batch
#             x, y = x.to(DEVICE), y.to(DEVICE)
#             pred = model(x)

#             total_loss += loss_fn(pred, y).item()

#             pred_y_flat, y_flat = binary_one_hot_output_transform(pred, y)
#             con_mat = confusion_matrix(y_flat, pred_y_flat, labels=[i for i in range(out_channels)])
#             final_con_mat.append(con_mat)
    

#     final_con_mat = np.sum(final_con_mat, axis = 0)

#     test_loss = total_loss / len(dataloader)
#     print(f"Testing Loss: {test_loss:>7f}")

#     print("\nConfusion Matrix\n", final_con_mat)

#     return final_con_mat


# def binary_one_hot_output_transform(pred_y, y):

#     pred_y = torch.argmax(pred_y, dim = 1).long().flatten().cpu()

#     y = y.flatten().cpu()

#     return pred_y, y


# def show_split_pairs(splits, split="train", n=6, seed=0):
#     x = splits[split]["input"]
#     z = splits[split]["mask"]

#     assert len(x) == len(z), "Input and mask length mismatch"

#     rng = np.random.default_rng(seed)
#     idxs = rng.choice(len(x), size=min(n, len(x)), replace=False)

#     for i in idxs:
#         xi = x[i][0]  # FIRST CHANNEL ONLY
#         zi = z[i][0]  # FIRST CHANNEL ONLY

#         fig, ax = plt.subplots(1, 3, figsize=(9, 3))

#         ax[0].imshow(xi, cmap="gray")
#         ax[0].set_title(f"{split} input")
#         ax[0].axis("off")

#         ax[1].imshow(zi, cmap="gray")
#         ax[1].set_title("mask")
#         ax[1].axis("off")

#         # overlay
#         ax[2].imshow(xi, cmap="gray")
#         ax[2].imshow(zi, alpha=0.4)
#         ax[2].set_title("overlay")
#         ax[2].axis("off")

#         plt.tight_layout()
#         plt.show()


# def model_validation_preview(dataloader, model, odirname):
#     count = 0

#     odirname = Path(f"{odirname}/Pred_Showcase")
#     odirname.mkdir(parents=True, exist_ok=True)

#     model.eval()
#     with torch.no_grad():
#         for batch in dataloader:
#             x, y = batch
#             x, y = x.to(DEVICE), y.to(DEVICE)
#             pred = model(x)

#             y = y.squeeze(-3)

#             for i in range(y.shape[0]):
#                 x0 = x[i].detach().cpu().numpy()
#                 y0 = y[i].detach().cpu().numpy()
#                 p0 = torch.argmax(pred, dim = 1)[i].detach().cpu().numpy().astype(np.uint16)

#                 if x0.ndim == 4:
#                     x0 = np.max(x0, axis = 1)
#                     y0 = np.max(y0, axis = 1).squeeze()
#                     p0 = np.max(p0, axis = 0)
                
#                 plt.figure(figsize=(12,4))

#                 plt.subplot(1,3,1)
#                 # show first channel of input (adjust if you want RGB etc.)
#                 plt.imshow(x0[0], cmap="gray")
#                 plt.title("Input")
#                 plt.axis("off")

#                 plt.subplot(1,3,2)
#                 plt.imshow(y0, cmap="gray")
#                 plt.title("Ground truth")
#                 plt.axis("off")

#                 plt.subplot(1,3,3)
#                 plt.imshow(p0, cmap="gray")
#                 plt.title("Pred mask")
#                 plt.axis("off")

#                 plt.savefig(
#                     f"{odirname}/Pred_{count}.png",
#                     dpi=300,
#                     bbox_inches="tight",
#                 )
#                 plt.close()
#                 count += 1


# def save_run_config(output_prefix, loss_fn, optimizer, model):
#     config = {
#         name: value
#         for name, value in globals().items()
#         if name.isupper() and not name.startswith("__")
#     }
    
#     config["loss_fn"] = loss_fn.__class__.__name__
#     config["optimizer"] = optimizer.__class__.__name__
#     config["model"] = model.__class__.__name__


#     config_path = Path(f"{output_prefix}_config.json")
#     with open(config_path, "w") as f:
#         json.dump(config, f, indent=4, default=str)

# def main(image_dir, 
#         mask_img_dir,
#         split_train,
#         split_validation,
#         split_test,
#         generate_data,
#         num_images,
#         loss_function,
#         optimizer,
#         learning_rate,
#         batch_size,
#         num_epochs,
#         input_channels,
#         feature_channels,
#         output_channels,
#         num_blocks,
#         use_batchnorm,
#         model_odirname,
#         seed
#         ):

#     # TODO: Maybe instead of loading everything into the RAM, get the paths and then 
#     # inside the dataset class load each pair of input and mask ?
#     input_images = load_images(image_dir)
#     mask_images = load_images(mask_img_dir, is_mask = True)

#     seed_everything(seed)# seed = np.random.default_rng().integers(0, 1_000_000) # TODO: MAKE THE SPLIT IN TORCH NOT NP
#     splits = ordered_split(input_images, mask_images, split_train, split_validation, split_test, 0.7, 0.3) # TODO: MAKE SURE THAT THE BALANCE IS BASED ON THE PROPORTION FROM THE LISTS RATHER THAN RECEIVING THE INPUT

#     # show_split_pairs(splits, split="train", n=210, seed=1)
#     # show_split_pairs(splits, split="val", n=46, seed=2)
#     # show_split_pairs(splits, split="test", n=44, seed=3)

#     train_dataset = BacteriaMaskPair(splits["train"]["input"], splits["train"]["mask"], transform=transform3d, generate_data = generate_data, num_images = num_images)
#     validation_dataset = BacteriaMaskPair(splits["val"]["input"], splits["val"]["mask"], transform=ComposePair([DepthPadding((2, 1)), CenterCrop(SIZE)]))
#     test_dataset  = BacteriaMaskPair(splits["test"]["input"],  splits["test"]["mask"], transform=ComposePair([DepthPadding((2, 1)), CenterCrop(SIZE)]))
     
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
#     test_dataloader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     print("Train Dataset: ", len(train_dataloader.dataset), " Images")
#     print("Validation Dataset: ", len(validation_dataloader.dataset), " Images")
#     print("Test Dataset: ", len(test_dataloader.dataset), " Images")
#     print()

#     model = UNet3D(input_channels, output_channels, feature_channels, num_blocks = num_blocks, use_batchnorm = use_batchnorm).to(DEVICE)

#     class_weights = compute_class_weights_from_dataloader(train_dataloader, output_channels)
#     print(class_weights)
#     exit()
#     loss_fn = DiceWeightedCrossEntropyLoss(0.3, class_weights) # CrossEntropyLoss(weight=class_weights) #SoftDiceLoss() # FocalLoss(gamma=1.0)
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # TODO: IMPROVE PARAMETERS

#     train_losses = list()
#     validation_losses = list()
#     for epoch in range(num_epochs):
#         print(f"Epoch {epoch+1}\n-------------------------------")

#         if not epoch % 100 and epoch != 0:
#             lr = optimizer.param_groups[0]['lr']
#             optimizer.param_groups[0]['lr'] = lr / 10

#         train_loss = train_model(train_dataloader, model, loss_fn, optimizer)
#         train_losses.append(train_loss)
        
#         validation_loss = validate_model(validation_dataloader, model, loss_fn)
#         validation_losses.append(validation_loss)
    
#     conf_matrix = test_model(test_dataloader, model, loss_fn, output_channels)
    
#     odirname = Path(model_odirname).parent
#     odirname.mkdir(parents=True, exist_ok=True)

#     torch.save(model.state_dict(), f"{model_odirname}.pt")
#     save_run_config(model_odirname, loss_fn, optimizer, model)


#     model_validation_preview(test_dataloader, model, odirname)
#     plot_confusion_matrix(conf_matrix, odirname)
#     plot_train_validation_loss(train_losses, validation_losses, odirname)


# if __name__ == "__main__":
#     main(IMAGE_DIR,
#         MASK_IMG_DIR, 
#         SPLIT_TRAIN,
#         SPLIT_VALIDATION,
#         SPLIT_TEST,
#         GENERATE_DATA,
#         NUM_IMAGES,
#         LOSS_FUNCTION,
#         OPTIMIZER,
#         LEARNING_RATE,
#         BATCH_SIZE,
#         NUM_EPOCHS,
#         INPUT_CHANNELS,
#         FEATURE_CHANNELS,
#         OUTPUT_CHANNELS,
#         NUM_BLOCKS,
#         USE_BATCHNORM,
#         MODEL_ODIRNAME,
#         SEED)

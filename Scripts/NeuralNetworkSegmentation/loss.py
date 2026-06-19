from monai.losses import (
    DiceLoss,
    DiceCELoss,
    FocalLoss,
    DiceFocalLoss,
)

# UNet model has a final activation layer that converts logits into probabilities using softmax
# NOTE: In near future handle missing activation function or just have to be careful when building the config file 

def get_loss(loss_config, weights = None):
    loss_name = loss_config["name"]

    if loss_name == "DiceLoss":
        smooth = loss_config.get("smooth", 1)
        incude_bg = loss_config.get("include_background", True)
        return DiceLoss(
            to_onehot_y=True,
            include_background=incude_bg,
            smooth_dr=smooth,
            smooth_nr=smooth,
        )
    
    elif loss_name == "DiceCELoss": # problem with passing weights for now
        smooth = loss_config.get("smooth", 1)
        incude_bg = loss_config.get("include_background", True)
        lambda_dice = loss_config.get("lambda_dice", 0.5)
        lambda_ce = loss_config.get("lambda_ce", 0.5)
        return DiceCELoss(
            to_onehot_y=True,
            include_background=incude_bg,
            smooth_dr=smooth,
            smooth_nr=smooth,
            weight=weights,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
        )
    
    elif loss_name == "FocalLoss":
        gamma = loss_config.get("gamma", 1)
        incude_bg = loss_config.get("include_background", True)
        return FocalLoss(
            to_onehot_y=True,
            gamma=gamma,
            include_background=incude_bg,
        )
    
    elif loss_name == "DiceFocalLoss":
        smooth = loss_config.get("smooth", 1)
        incude_bg = loss_config.get("include_background", True)
        gamma = loss_config.get("gamma", 1)
        lambda_dice = loss_config.get("lambda_dice", 0.5)
        lambda_focal = loss_config.get("lambda_focal", 0.5)


        return DiceFocalLoss(
            to_onehot_y=True,
            include_background=incude_bg,
            gamma=gamma,
            smooth_dr=smooth,
            smooth_nr=smooth,
            lambda_dice=lambda_dice,
            lambda_focal=lambda_focal,
        )
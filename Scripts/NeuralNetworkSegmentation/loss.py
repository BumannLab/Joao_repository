from monai.losses import (
    DiceLoss,
    DiceCELoss,
    FocalLoss,
    DiceFocalLoss,
)

# UNet model has a final activation layer that converts logits into probabilities using softmax
# NOTE: In near future handle missing activation function or just have to be careful when building the config file 


# Most of these losses require onehot encoding when using softmax;
# If final activation is softmax use the OneHot

def get_loss(loss_config, weights = None):
    loss_name = loss_config["name"]
    incude_bg = loss_config.get("include_background", True)
    
    sigmoid = loss_config.get("sigmoid", False)
    softmax = loss_config.get("softmax", False)
        
    if all(function is False for function in [sigmoid, softmax]):
        raise ValueError("No functions (sigmoid and softmax) were found in the loss section of" \
        " the config. Please add either one")

    if all(function is True for function in [sigmoid, softmax]):
        raise ValueError("Both functions (sigmoid and softmax) were found in the loss section of" \
        " the config. Only one can be selected")

    if loss_name == "DiceLoss":
        smooth = loss_config.get("smooth", 1)
        return DiceLoss(
            sigmoid=sigmoid,
            softmax=softmax, 
            include_background=incude_bg,
            smooth_dr=smooth,
            smooth_nr=smooth,
        )
    
    elif loss_name == "DiceCELoss": # problem with passing weights for now
        smooth = loss_config.get("smooth", 1)
        lambda_dice = loss_config.get("lambda_dice", 0.5)
        lambda_ce = loss_config.get("lambda_ce", 0.5)
        return DiceCELoss(
            sigmoid=sigmoid,
            softmax=softmax,
            include_background=incude_bg,
            smooth_dr=smooth,
            smooth_nr=smooth,
            weight=weights,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
        )
    
    elif loss_name == "FocalLoss": # 
        gamma = loss_config.get("gamma", 1)
        return FocalLoss(
            use_softmax=softmax if softmax else sigmoid,
            gamma=gamma,
            include_background=incude_bg,
        )
    
    elif loss_name == "DiceFocalLoss":
        smooth_dr = loss_config.get("smooth_dr", 0.00001)
        smooth_nr = loss_config.get("smooth_nr", 0.00001)
        gamma = loss_config.get("gamma", 1)
        lambda_dice = loss_config.get("lambda_dice", 0.5)
        lambda_focal = loss_config.get("lambda_focal", 0.5)
        return DiceFocalLoss(
            sigmoid=sigmoid,
            softmax=softmax,
            include_background=incude_bg,
            gamma=gamma,
            smooth_dr=smooth_dr,
            smooth_nr=smooth_nr,
            lambda_dice=lambda_dice,
            lambda_focal=lambda_focal,
        )
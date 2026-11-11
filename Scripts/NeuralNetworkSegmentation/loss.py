from monai.losses import (
    DiceLoss,
    DiceCELoss,
    FocalLoss,
    DiceFocalLoss,
)

# The model returns logits prior to sigmoid activation

def get_loss(loss_config, weights = None):
    
    loss_name = loss_config["name"]
    
    sigmoid = True

    if loss_name == "DiceLoss":
        smooth_nr = loss_config.get("smooth_nr", 0.00001)
        smooth_dr = loss_config.get("smooth_dr", 0.00001)
        return DiceLoss(
            sigmoid=sigmoid,
            smooth_dr=smooth_dr,
            smooth_nr=smooth_nr,
        )
    
    elif loss_name == "DiceCELoss": # problem with passing weights for now
        smooth_nr = loss_config.get("smooth_nr", 0.00001)
        smooth_dr = loss_config.get("smooth_dr", 0.00001)
        lambda_dice = loss_config.get("lambda_dice", 0.5)
        lambda_ce = loss_config.get("lambda_ce", 0.5)
        return DiceCELoss(
            sigmoid=sigmoid,
            smooth_dr=smooth_dr,
            smooth_nr=smooth_nr,
            weight=weights,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
        )
    
    elif loss_name == "FocalLoss": # 
        gamma = loss_config.get("gamma", 1)
        return FocalLoss(
            use_softmax=False, # Use sigmoid
            gamma=gamma,
        )
    
    elif loss_name == "DiceFocalLoss":
        smooth_dr = loss_config.get("smooth_dr", 0.00001)
        smooth_nr = loss_config.get("smooth_nr", 0.00001)
        gamma = loss_config.get("gamma", 1)
        lambda_dice = loss_config.get("lambda_dice", 0.5)
        lambda_focal = loss_config.get("lambda_focal", 0.5)
        return DiceFocalLoss(
            sigmoid=sigmoid,
            gamma=gamma,
            smooth_dr=smooth_dr,
            smooth_nr=smooth_nr,
            lambda_dice=lambda_dice,
            lambda_focal=lambda_focal,
        )
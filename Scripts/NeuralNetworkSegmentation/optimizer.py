import torch.optim as optim


def get_optimizer(optimizer_config, model):
    optimizer_name = optimizer_config["name"]

    learning_rate = optimizer_config.get("lr", 0.0001)
    weight_decay = optimizer_config.get("weight_decay", 0)

    if optimizer_name == "Adam":
        betas = optimizer_config.get("betas", (0.9, 0.999))
        return optim.Adam(model.parameters(),
                          lr=learning_rate,
                          weight_decay=weight_decay,
                          betas=betas
        )
    elif optimizer_name == "SGD":
        momentum = optimizer_config.get("momentum", 0)
        dampening = optimizer_config.get("dampening", 0)
        nesterov = optimizer_config.get("nesterov", False)
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            dampening=dampening,
            nesterov=nesterov,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_lr_scheduler(scheduler_config, optimizer): # Handle none
    scheduler_name = scheduler_config.get("name", {})

    if scheduler_name == "StepLR":
        step_size = scheduler_config.get("step_size", 10)
        gamma = scheduler_config.get("gamma", 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    

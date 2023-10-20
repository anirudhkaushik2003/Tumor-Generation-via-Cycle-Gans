import wandb

def wandb_init(learning_rateG, learning_rateD, epochs, batch_size, patch_size, architechture, dataset, multiGPU=False):
    project = "Tumor-Generation"
    config = {
        "learning_rateG": learning_rateG,
        "learning_rateD": learning_rateD,
        "architecture": architechture,
        "dataset": dataset,
        "epochs": epochs,
        "batch_size": batch_size,
        "patch_size": patch_size,
        "multiGPU": multiGPU
    }

    wandb.init(project=project, config=config)


def log_images(healthy, tumor, epoch, real=False):
    if real:
        images = wandb.Image(
            [healthy, tumor],
            caption=["Healthy (real)", "Tumor (real)"]
        )
    else:
        images = wandb.Image(
            [healthy, tumor],
            caption=["Healthy (generated)", "Tumor (generated)"]
        )

    wandb.log({f"Epoch {epoch+1}": images})
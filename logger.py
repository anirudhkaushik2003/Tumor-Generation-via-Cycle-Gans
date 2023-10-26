import wandb
import torchvision.utils as vutils
import torch 

def wandb_init(proj_name, run_name, learning_rateG, learning_rateD, epochs, batch_size, patch_size, architechture, dataset, multiGPU=False):
    project = proj_name
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

    wandb.init(project=project, config=config, name=run_name)


def log_images(real_healthy, real_tumor, healthy, tumor, epoch):
    images_array = vutils.make_grid(torch.cat((real_healthy, healthy, real_tumor, tumor), dim=0 ), padding=0, normalize=True, nrow=2, scale_each=False)
    images = wandb.Image(
        images_array,
        caption="Real Healthy, Generated Tumor, Real Tumor, Generated Healthy"
    )

    wandb.log({f"Results": images})
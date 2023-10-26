import argparse


class TrainOptions:
    def __init__(self):
        pass


    def initialize(self):
        parser = argparse.ArgumentParser(
            prog="main.py",
            description="Train a cycleGAN model",
            epilog="Currently adds a tumor to healthy brain MRIs",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parameters = parser.add_argument_group("Parameters")
        parameters.add_argument(
            "--epochs",
            type=int,
            default=100,
            help="Number of epochs to train for before linear lr decay for same number of epochs",
        )
        parameters.add_argument(
            "--patch_shape", type=int, default=-1, help="Patch shape for training"
        )
        parameters.add_argument(
            "--batch_size", type=int, default=1, help="Batch size for training"
        )
        parameters.add_argument(
            "--image_size", type=int, default=256, help="Image size for training"
        )
        parameters.add_argument(
            "--image_channels", type=int, default=1, help="Image channels for training"
        )
        parameters.add_argument(
            "--lambda_1", type=int, default=5, help="Lambda 1 for training"
        )
        parameters.add_argument(
            "--lambda_2", type=int, default=10, help="Lambda 2 for training"
        )
        parameters.add_argument(
            "--learning_rateG",
            type=float,
            default=0.0002,
            help="Learning rate for generator",
        )
        parameters.add_argument(
            "--learning_rateD",
            type=float,
            default=0.0002,
            help="Learning rate for discriminator",
        )
        parameters.add_argument(
            "--criterion", type=str, default="MSE", help="Loss function for criterionGAN"
        )
        parameters.add_argument(
            "--device", type=str, default="cuda:0", help="Device to train on"
        )
        parameters.add_argument(
            "--multiGPU", type=bool, default=False, help="Whether to use multiple GPUs"
        )
        parameters.add_argument(
            "--id", type=bool, default=True, help="Whether to use identity loss"
        )

        architecture = parser.add_argument_group("Architecture")
        architecture.add_argument(
            "--n_residual_blocks",
            type=int,
            default=9,
            help="Number of residual blocks in generator",
        )
        architecture.add_argument(
            "--disc_type", type=str, default="patchGAN", help="Type of discriminator to use"
        )

        logging = parser.add_argument_group("Logging")
        logging.add_argument(
            "--wandb", type=bool, default=False, help="Whether to use wandb"
        )
        logging.add_argument(
            "--wandb_project", type=str, default="Tumor-Generation", help="Wandb project name"
        )
        logging.add_argument(
            "--save_name", type=str, default="v1", help="Name of model to save"
        )
        logging.add_argument(
            "--run_name", type=str, default="vannila", help="Name of run to save"
        )

        return parser




#!/bin/bash

python main.py --disc_type UNET --device cuda:0 --criterion BCEwithLogits --save_name v1 --lambda_1 5 --lamda_2 10 --id True  --wandb True --run_name "UNET with BCE"
python main.py --disc_type UNET --device cuda:1 --criterion BCEwithLogits --save_name v2 --lambda_1 5 --lamda_2 25 --id True --wandb True --run_name "UNET with BCE l2=25"
python main.py --disc_type UNET --device cuda:2 --criterion BCEwithLogits --save_name v3 --lambda_1 5 --lamda_2 10 --id False --wandb True --run_name "UNET with BCE no id"
python main.py --disc_type patchGAN --device cuda:3 --criterion MSE --save_name v4 --lambda_1 5 --lamda_2 10 --id True --wandb True --run_name "PatchGAN with MSE"

#!/bin/bash
#SBATCH --job-name=calibrate
#SBATCH --mail-type=END,FAIL          
#SBATCH --mail-user=skylastolte4444@ufl.edu     	
#SBATCH --ntasks=1                   
#SBATCH --cpus-per-task=4
#SBATCH --distribution=block:block 
###SBATCH --partition=hpg-b200
#SBATCH --partition=hpg-turin
###SBATCH --constraint=a100 	 
#SBATCH --gres=gpu:1 	 
#SBATCH --mem=30gb                     
#SBATCH --time=48:00:00    
#SBATCH --account=camctrp
#SBATCH --qos=camctrp 
#SBATCH --output=test_%j.log 

module load conda

#conda create -n torchdomino python=3.9 -y
conda activate torchdomino

#conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
#conda install numpy pandas scikit-learn matplotlib pillow -y
#pip install torchmetrics
#pip install opencv-python tqdm
#pip install seaborn scipy
#pip install ttach

python train-Copy2.py --data_dir '/red/ruogu.fang/skylar/AFRL/Diffusion/Data/cifar10/' --output_dir '/red/ruogu.fang/skylar/AFRL/SAR_for_Uncertainty-main/SAR_for_Uncertainty-main/' --model_save_name resnet50_cifar10_domino_multiply_hc --dataset_name 'cifar10' --model_version 'resnet50' --use_DOMINO_multiply --matrix_csv '/red/ruogu.fang/skylar/AFRL/SAR_for_Uncertainty-main/SAR_for_Uncertainty-main/scripts/cifar10_hc.csv'

conda deactivate
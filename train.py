# coding=utf-8

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train and validate model.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Overarching path to save models + results.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--model_save_name", type=str, default="resnet50_CE", help="Model architecture name.")
    parser.add_argument("--model_version", type=str, default="resnet50", help="Which model architecture to use.")
    parser.add_argument("--IMG_CH", type=int, default=3, help="Number of Channels (usually 3).")
    parser.add_argument("--IMG_SIZE", type=int, default=128, help="Size desired for Images to be Cropped.")
    parser.add_argument("--dataset_name", type=str, default='sneakers', help="Which dataset are you running?")
    parser.add_argument('--use_DOMINO', action='store_true', help='use DOMINIO loss')
    parser.add_argument('--use_DOMINO_multiply', action='store_true', help='use DOMINIO multiply loss')
    parser.add_argument("--matrix_csv", type=str, help="csv for matrix penalty if a domino loss version is used.")
    parser.add_argument('--pick_a_and_b', action='store_true', help='pick the weight balances for CE and DOMINO (only with DOMINO)')
    
    return parser.parse_args()

args = parse_args()

#args.data_dir = '/red/ruogu.fang/skylar/AFRL/Diffusion/Data/sneakers-dataset/sneakers-dataset/'
#args.output_dir = '/red/ruogu.fang/skylar/AFRL/SAR_for_Uncertainty-main/SAR_for_Uncertainty-main/'
#batch_size = 100
#IMG_CH = 3
#IMG_SIZE = 128
#args.model_save_name
#DOMINO = args.use_DOMINO
#DOMINOM = args.use_DOMINO_multiply
#args.matrix_csv = '/red/ruogu.fang/skylar/AFRL/Diffusion/sneakers_hc.csv'
#args.num_epochs = 20
#pick_a_and_b:

##########################################################################################################################################################################

#torch
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, TensorDataset

#general
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mat_datafuncs
from PIL import Image
import re

import random
random.seed(1)
random_seed = random.seed(1)

#new loss
from DominoLoss import DOMINO_Loss
from DominoLossM import DOMINO_Loss_M

#results and calibration
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.calibration import calibration_curve
from torchmetrics.classification import MulticlassCalibrationError
from sklearn.metrics import brier_score_loss, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from reliability_diagrams import *

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from matplotlib.backends.backend_pdf import PdfPages

dataset_name = args.dataset_name.lower()
if dataset_name == "altpets":
    from dataloader_altpet import get_AltPet_loader
    
##########################################################################################################################################################################

#IMAGENET_NORMALIZATION_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
IMAGENET_NORMALIZATION_STATS = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def get_inv_normalize():
    tfms = T.Normalize(   mean= [-m/s for m, s in zip(IMAGENET_NORMALIZATION_STATS[0], IMAGENET_NORMALIZATION_STATS[1])],
                       std= [1/s for s in IMAGENET_NORMALIZATION_STATS[1]])
    return tfms
    
denorm = get_inv_normalize()

data_dir = args.data_dir

data_transform = transforms.Compose([
    #squarecrop,
    transforms.CenterCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Assuming images are in range [0, 1]
])

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.class_to_idx = {}
        self.classes = []

        pattern = re.compile(r'\.jpg$')
        for idx, cls_name in enumerate(sorted(os.listdir(data_dir))):
            if cls_name.startswith('.'):  # ✅ SKIP hidden directories
                continue
            class_dir = os.path.join(data_dir, cls_name)
            if not os.path.isdir(class_dir):
                continue

            valid_images = [
                os.path.join(class_dir, fname)
                for fname in os.listdir(class_dir)
                if pattern.search(fname)
            ]

            if valid_images:
                self.class_to_idx[cls_name] = len(self.classes)
                self.classes.append(cls_name)
                self.images.extend([(img_path, self.class_to_idx[cls_name]) for img_path in valid_images])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
    
# Create custom dataset
#train_dataset = CustomImageDataset(data_dir, transform=data_transform)

# Define batch size
batch_size = args.batch_size

#dataset_name = args.dataset_name.lower()

if dataset_name == "sneakers":
    print("sneakers")
    train_dataset = CustomImageDataset(data_dir, transform=data_transform)
    #class_names = dataset.classes
    
    class_names = sorted([d for d in os.listdir(args.data_dir)
    if not d.startswith('.') and d != '.ipynb_checkpoints' and os.path.isdir(os.path.join(args.data_dir, d))])
    
if dataset_name == "cars":
    print("Stanford Cars")
    train_dataset = CustomImageDataset(data_dir, transform=data_transform)
    #class_names = dataset.classes
    
    class_names = sorted([d for d in os.listdir(args.data_dir)
    if not d.startswith('.') and d != '.ipynb_checkpoints' and os.path.isdir(os.path.join(args.data_dir, d))])

elif dataset_name == "cifar10":
    print("CIFAR10")
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=data_transform)
    class_names = train_dataset.classes
    
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=data_transform)
    
elif dataset_name == "cifar100":
    print("CIFAR100")
    train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=data_transform)
    class_names = train_dataset.classes
    
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=data_transform)

elif dataset_name == "mnist":
    print("MNIST")
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert 1→3 channels
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]))
    class_names = [str(i) for i in range(10)]
    
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert 1→3 channels
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]))
    
elif dataset_name == "mstar":
    
    print("MSTAR")
    
    def DataFromDirectories(data_dir):
        data_transforms = {
        'train': transforms.Compose([
            transforms.CenterCrop(64),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop(64),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]),
        'mat_val': transforms.Compose([
            transforms.CenterCrop(64),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]),
        'mat_val_recon': transforms.Compose([
            transforms.CenterCrop(64),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]),    
        }
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                        for x in ['train', 'test', 'mat_val', 'mat_val_recon']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=100,
                                                    shuffle=True, num_workers=4)
                    for x in ['train', 'test', 'mat_val', 'mat_val_recon']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test', 'mat_val', 'mat_val_recon']}
        class_names = image_datasets['train'].classes
        print(class_names)

        return dataloaders, dataset_sizes, class_names
    
    dataloaders, dataset_sizes, class_names = DataFromDirectories(data_dir)
    
    train_loader = dataloaders['train']
    valid_loader = dataloaders['mat_val']
    matrix_loader = dataloaders['mat_val_recon']
    test_loader = dataloaders['test']
    
    train_size = dataset_sizes['train']
    val_size = dataset_sizes['mat_val']
    matrix_size = dataset_sizes['mat_val_recon']
    test_size = dataset_sizes['test']
    
elif dataset_name == "altpets":
    print("Alternative Dataset of Pet Data")

    # set up working data location
    #data_root = '/blue/ruogu.fang/skylastolte4444/Airplanes/SAR_for_Uncertainty-main/SAR_for_Uncertainty-main/AltPets'

    train_loader = get_AltPet_loader(data_dir + "/abs_train_filenames.txt", batch_size, augment=True, shuffle=True, num_workers=1, pin_memory=True)
    valid_loader = get_AltPet_loader(data_dir + "/abs_valid_filenames.txt", batch_size, augment=False, shuffle=True, num_workers=1, pin_memory=True)
    matrix_loader = get_AltPet_loader(data_dir + "/abs_matrix_filenames.txt", batch_size, augment=False, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = get_AltPet_loader(data_dir + "/abs_test_filenames.txt", batch_size, augment=False, shuffle=True, num_workers=1, pin_memory=True)

    #classes
    with open(os.path.join(data_dir, 'folder_names.txt'), 'r') as f:
        class_names = f.read().splitlines()

    object_dict = {}
    for o in range(len(class_names)):
        object_dict[class_names[o]] = o
        
else:
    raise ValueError(f"Unknown dataset: {args.dataset_name}. Supported: sneakers, cifar10, mnist, altpets, mstar, cars")

if dataset_name != "altpets" and dataset_name != "mstar": # and dataset_name != "cars":
    
    # Define dataset split lengths
    total_size = len(train_dataset)
    train_size = int(0.8 * total_size)
    val_size = test_size = (total_size - train_size) // 2
    remainder = total_size - (train_size + val_size + test_size)
    train_size += remainder  

    # Perform the split
    train_set, val_set, test_set = random_split(
        train_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  
    )
    #train_set, val_set = random_split(
    #    train_dataset, [train_size, val_size],
    #    generator=torch.Generator().manual_seed(42)  
    #)
    
    if dataset_name != "sneakers" and dataset_name != "cars":
        val_set = val_set + test_set
        test_set = test_dataset

if dataset_name == "sneakers" or dataset_name == "cars":
    test_filenames = [train_dataset.images[i][0] for i in test_set.indices]

if dataset_name != "altpets" and dataset_name != "mstar": # and dataset_name != "cars":    
    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Class info
#class_names = sorted(os.listdir(args.data_dir))

N_CLASSES = len(class_names)
IMG_CH = args.IMG_CH
IMG_SIZE = args.IMG_SIZE

print(f"Classes: {N_CLASSES}")
if dataset_name != "altpets" and dataset_name != "mstar" and dataset_name != "cars":
    print(f"Train size: {len(train_set)}, Val size: {len(val_set)}, Test size: {len(test_set)}")

print(len(class_names))
print(class_names)

##########################################################################################################################################################################

#add directories for models and results

#generic directories
model_save_path = args.output_dir + args.dataset_name + '/models/'
results_save_path = args.output_dir + args.dataset_name + '/results/'

#specific model definitions
model_name = args.model_save_name #'resnet50_domino_chooseab4'#DOMINO-SSIM_REAL'
DOMINO = args.use_DOMINO
DOMINOM = args.use_DOMINO_multiply

#specific model path
results_model = results_save_path + model_name

#make all non-existing directories
isModelPath = os.path.exists(model_save_path)
isResultsPath = os.path.exists(results_save_path)
isModelResultsPath = os.path.exists(results_model)

if not isModelPath:
    os.makedirs(model_save_path)
    
if not isResultsPath:
    os.makedirs(results_save_path)

if not isModelResultsPath:
    os.makedirs(results_model)

#os.makedirs(results_save_path + model_name

if DOMINO or DOMINOM:
    matrix_vals = pd.read_csv(args.matrix_csv, index_col=0, header=0)#confusionmatrix_norm_Kyle.csv', index_col = 0, header = 0)#matrix_vals = pd.read_csv('/red/ruogu.fang/skylar/Airplanes/Diffusion/matrix_sneakers_900.csv', index_col=None, header=None)#confusionmatrix_norm_Kyle.csv', index_col = 0, header = 0)
    #matrix_vals = pd.read_csv('/red/ruogu.fang/skylar/Airplanes/Diffusion/matrix_sneakers_900.csv', index_col=None, header=None)#confusionmatrix_norm_Kyle.csv', index_col = 0, header = 0)
    matrix_penalty = 3.0 * torch.from_numpy(matrix_vals.to_numpy())
    matrix_penalty = matrix_penalty.float().cuda()
    print(matrix_penalty.shape)
    
if DOMINO:
    a = 0.5
    b = 0.5
    
    print('DOMINO is working')
    
if DOMINOM:
    print('DOMINO Multiply is working')

# load a pre-trained model
model_version = args.model_version

if model_version == "resnet18":
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, len(class_names))#len(object_categories))
    print('Using RESNET 18')
elif model_version == "resnet50":
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(2048, len(class_names))
    print('Using RESNET 50')
elif model_version == "resnet101":
    model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    model.fc = nn.Linear(2048, len(class_names))
    print('Using RESNET 101')
elif model_version == "densenet121":
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    
    # DenseNet puts the final layer in `classifier`
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, len(class_names))
    print('Using DesNet 121')
elif model_version == "DINO":
    backbone = torch.hub.load('facebookresearch/dino:main','dino_vits16')

    backbone.head = nn.Identity()

    class DinoClassifier(nn.Module):
        def __init__(self, backbone, num_classes):
            super().__init__()
            self.backbone = backbone
            self.fc = nn.Linear(384, num_classes)  # ViT-S = 384

        def forward(self, x):
            feats = self.backbone(x)
            return self.fc(feats)

    model = DinoClassifier(backbone, len(class_names))
    
    for p in model.backbone.parameters():
        p.requires_grad = False
    
    print("Using DINO ViT-S/16")
elif model_version == "DINOv3":
    
    weights = "/blue/ruogu.fang/skylastolte4444/skylar/AFRL/SAR_for_Uncertainty-main/SAR_for_Uncertainty-main/scripts/dino_models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    
    DINOV3_GITHUB_LOCATION = "dinov3"

    if os.getenv("DINOV3_LOCATION") is not None:
        DINOV3_LOCATION = os.getenv("DINOV3_LOCATION")
    else:
        DINOV3_LOCATION = DINOV3_GITHUB_LOCATION
    
    print(f"DINOv3 location set to {DINOV3_LOCATION}")

    # examples of available DINOv3 models:
    MODEL_DINOV3_VITS = "dinov3_vits16"
    MODEL_DINOV3_VITSP = "dinov3_vits16plus"
    MODEL_DINOV3_VITB = "dinov3_vitb16"
    MODEL_DINOV3_VITL = "dinov3_vitl16"
    MODEL_DINOV3_VITHP = "dinov3_vith16plus"
    MODEL_DINOV3_VIT7B = "dinov3_vit7b16"

    MODEL_DINOV3_CONVS = "dinov3_convnext_small"
    MODEL_DINOV3_CONVL = "dinov3_convnext_large"

    MODEL_NAME = MODEL_DINOV3_VITB

    MODEL_TO_NUM_LAYERS = {
        MODEL_DINOV3_VITS: 12,
        MODEL_DINOV3_VITSP: 12,
        MODEL_DINOV3_VITB: 12,
        MODEL_DINOV3_VITL: 24,
        MODEL_DINOV3_VITHP: 32,
        MODEL_DINOV3_VIT7B: 40,
        MODEL_DINOV3_CONVS: 12,
        MODEL_DINOV3_CONVL: 12
    }

    n_layers = MODEL_TO_NUM_LAYERS[MODEL_NAME]

    model_backbone = torch.hub.load(
        repo_or_dir=DINOV3_LOCATION,
        model=MODEL_NAME,
        source="local",
        #weights="/home/kyle/Desktop/Misc/dino/weights/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth"
        #weights="/home/kyle/Desktop/Misc/dino/weights/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth"
        #weights="/home/kyle/Desktop/Misc/dino/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
        weights=weights
    )
    
    class DinoClassifier(nn.Module):
        def __init__(self, backbone, num_classes):
            super().__init__()
            self.backbone = backbone
            self.fc = nn.Linear(768, num_classes)  # ViT-S = 384

        def forward(self, x):
            feats = self.backbone(x)
            return self.fc(feats)

    model = DinoClassifier(model_backbone, len(class_names))
    
    print("Using DINO v3 ViT-S/16")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Specify Loss
if DOMINO:
    criterion = DOMINO_Loss()
elif DOMINOM:
    criterion = DOMINO_Loss_M()
else:
    criterion = nn.CrossEntropyLoss()
    #criterion = FocalLoss2(gamma=2.0, alpha=None)

 # construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# epoch
num_epochs = args.epochs
logging_freq = 147
save_freq = 5

##########################################################################################################################################################################

metric1 = MulticlassCalibrationError(num_classes=len(class_names), n_bins=10, norm='l1')
#ECE = metric1(o,l)

if args.pick_a_and_b:

    model2 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model2.fc = nn.Linear(2048, len(class_names))
    model2.to(device)

    def train_with_ab(a, b, num_epochs=20):
        val_acc_best = 0
        ECE_best = 100
        for epoch in range(num_epochs):
            model2.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if inputs.shape[1] == 1:
                    inputs = inputs.repeat(1, 3, 1, 1).float()
            
                optimizer.zero_grad()
                outputs = model2(inputs)
                loss = criterion(outputs, labels, matrix_penalty, a, b)
                loss.backward()
                optimizer.step()
        
            # Validation
            model2.eval()
            val_correct, val_seen = 0, 0
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    if inputs.shape[1] == 1:
                        inputs = inputs.repeat(1, 3, 1, 1).float()
                    outputs = model2(inputs)
                    val_correct += (outputs.argmax(dim=1) == labels).float().sum().item()
                    val_seen += len(labels)
                
                    ECE = metric1(outputs,labels)
                
            val_acc = val_correct / val_seen
            val_acc_best = max(val_acc_best, val_acc)
        
            ECE_best = min(ECE_best, ECE)
        
        return ECE_best #val_acc_best


    if DOMINO:

        #best_acc, best_a, best_b = 0, None, None
        best_ECE, best_a, best_b = 100, None, None
    
        for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            #for b in [0.1, 0.5, 1.0]:
            for b in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                #b = 1 - a
                #acc = train_with_ab(a, b)
                ECE = train_with_ab(a,b)
                #if acc > best_acc:
                if ECE < best_ECE:
                    best_acc, best_a, best_b = ECE, a, b #acc, a, b
                print(f"a is {a} b is {b}", a, b)
        print(f"Best a={best_a}, b={best_b}, acc={best_acc}")

        a = best_a
        b = round(best_b,1)
    
        print(a)
        print(b)
    
##########################################################################################################################################################################

val_acc_best = 0
val_list = []
ece_list = []
#val_loss_total = []

train_list = []

for epoch in range(num_epochs):
    running_loss = 0.
    correct = 0.
    seen = 0.
    val_correct = 0.
    val_seen = 0.
    logging_step = 1
    
    for i, data in enumerate(train_loader, 0):##dataloaders['train'], 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        bs, c, h, w = inputs.shape
        if c == 1:
            inputs = inputs.repeat(1, 3, 1, 1).float()
            
        #if dataset_name == "cars":
        #    labels = labels.long().view(-1)

        optimizer.zero_grad()
        outputs = model(inputs)
        #outputs = outputs.logits
        if DOMINO:
            loss = criterion(outputs, labels, matrix_penalty, a, b)
        elif DOMINOM:
            loss = criterion(outputs, labels, matrix_penalty, 1)
        else:
            loss = criterion(outputs, labels)
            #loss = torchvision.ops.sigmoid_focal_loss(outputs, labels)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).float().sum()
        seen += len(labels)
        
    ece_running = 0.
    counter  = 0.
    #val_loss_section = 0.
    
    for i, data in enumerate(valid_loader, 0):##dataloaders['val'], 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        bs, c, h, w = inputs.shape
        if c == 1:
            inputs = inputs.repeat(1, 3, 1, 1).float()

        outputs = model(inputs)
    
        val_correct += (outputs.argmax(dim=1) == labels).float().sum()
        val_seen += len(labels)
        
        ece_running += metric1(outputs, labels)
        counter += 1
        
        #if DOMINO:
        #    val_loss = criterion(outputs, labels, matrix_penalty, a, b)
        #elif DOMINOM:
        #    val_loss = criterion(outputs, labels, matrix_penalty, 1)
        #else:
        #    val_loss = criterion(outputs, labels)
            #loss = torchvision.ops.sigmoid_focal_loss(outputs, labels)
            
        #val_loss_section+=val_loss
    
    ece_running = ece_running/counter
    ece_list.append(ece_running)
    
    #changed to only save models when validation improves
    val_acc = val_correct/val_seen
    val_list.append(val_acc)
    
    train_acc = correct/seen
    train_list.append(train_acc)
    
    #val_loss_total.append(val_loss_section)
    
    if val_acc>val_acc_best:
        torch.save(model.state_dict(), model_save_path + model_name + '.pth')
        val_acc_best=val_acc
        print('The new best validation accuracy is %.4f, saving model' % (val_acc_best))

    print("Epoch %d, loss: %.3f, Train acc: %.4f, Val acc: %.4f" % (epoch + 1,  running_loss/seen, correct/seen, val_correct/val_seen))

##########################################################################################################################################################################    
    
ece_list_cpu = [t.cpu() for t in ece_list]
val_list_cpu = [t.cpu() for t in val_list]
train_list_cpu = [t.cpu() for t in train_list]
#val_loss_total_cpu = [t.cpu() for t in val_loss_total]

epochs = list(range(0, num_epochs))

# Create a dictionary for DataFrame creation
data = {'Epochs': epochs, 'ECE': [t.item() for t in ece_list_cpu], 'ValAcc': [t.item() for t in val_list_cpu], 'TrainAcc': [t.item() for t in train_list_cpu]}
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv(os.path.join(results_model,'calibration_vs_accuracy.csv'), index=False)

# Plot the lists
plt.plot(epochs, ece_list_cpu, marker='o', linestyle='-', label='Calibration')
plt.plot(epochs, val_list_cpu, marker='o', linestyle='-', label='Validation Accuracy')
plt.plot(epochs, train_list_cpu, marker='o', linestyle='-', label='Train Accuracy')
#plt.plot(epochs, val_loss_total_cpu, label='Validation Loss')

# Add labels and a title for clarity
#plt.xlabel("Calibration")
#plt.ylabel("Accuracy")
plt.title("Calibration and Accuracy across Epochs")
plt.xlim(0, num_epochs)
plt.legend()
plt.xlabel("Epochs")

plt.yticks(np.arange(0.0, 1.01, 0.05))  # ticks every 0.05
plt.ylim(0, 1)

# Display the plot
#plt.show()
plt.savefig(os.path.join(results_model,'calibration_vs_accuracy.pdf'))

##########################################################################################################################################################################  

#reload best model
model.load_state_dict(torch.load(model_save_path + model_name + '.pth'))

#compute test using only the best performing model 
#(hopefully the following steps may be replaced by a testing dataset)

test_correct = 0.
test_seen = 0.
top2 = []
top3 = []

for i, data in enumerate(test_loader, 0):##dataloaders['val'], 0):
    
    torch.cuda.empty_cache()
    
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    #if dataset_name == "cars":
    #    labels = labels.long().view(-1)

    bs, c, h, w = inputs.shape
    #print(inputs.shape)
    if c == 1:
        inputs = inputs.repeat(1, 3, 1, 1).float()

    outputs = model(inputs)
    
    test_correct += (outputs.argmax(dim=1) == labels).float().sum()
    test_seen += len(labels)
    
    top2.append(top_k_accuracy_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy(), k=2, normalize=True, sample_weight=None, labels=list(range(N_CLASSES))))
    top3.append(top_k_accuracy_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy(), k=3, normalize=True, sample_weight=None, labels=list(range(N_CLASSES))))
    
    #save targets, predictions, and outputs for analysis
    if i==0:
        outputs_total = outputs.cpu().detach().numpy()
        #preds_total = outputs.argmax(dim=1)
        labels_total = labels.cpu().detach().numpy()
    else:
        outputs_total = np.concatenate((outputs_total, outputs.cpu().detach().numpy()), axis=0)
        #preds_total = torch.cat((preds_total, outputs.argmax(dim=1)), dim=0)
        labels_total = np.concatenate((labels_total, labels.cpu().detach().numpy()), axis=0)
        
    torch.cuda.empty_cache()
    
    #print(i)
    
    #if i > 945:
    #    break

preds_total = torch.Tensor(outputs_total).argmax(dim=1)
        
#verify sizes
print(outputs_total.shape)
print(preds_total.shape)
print(labels_total.shape)

#outputs_total = outputs_total.cpu().detach().numpy()
preds_total = preds_total.cpu().detach().numpy()
#labels_total = labels_total.cpu().detach().numpy()

print('The Top 1 accuracy on the testing set is: %.4f' % (test_correct/test_seen))
print('The Top 2 accuracy on the testing set is: %.4f' % (sum(top2)/len(top2)))
print('The Top 3 accuracy on the testing set is: %.4f' % (sum(top3)/len(top3)))

##########################################################################################################################################################################  

#confusion matrix on test data

#def plot_confusion_matrix(labels, pred_labels, classes):

#    fig = plt.figure(figsize=(10, 10))
 #   ax = fig.add_subplot(1, 1, 1)
 #   cm = confusion_matrix(labels, pred_labels)
 #   cm = ConfusionMatrixDisplay(cm, display_labels=classes)
 #   cm.plot(values_format='d', cmap='Blues', ax=ax)
 #   plt.grid(False)
#    plt.xticks(rotation=90)
    
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(labels, pred_labels, classes):
    num_classes = len(classes)
    label_indices = np.arange(num_classes)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    cm = confusion_matrix(
        labels,
        pred_labels,
        labels=label_indices   # <<< THIS IS THE FIX
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes
    )

    disp.plot(
        include_values=False,  # IMPORTANT for 196 classes
        cmap="Blues",
        ax=ax,
        xticks_rotation=90
    )

    ax.grid(False)


plot_confusion_matrix(labels_total, preds_total, class_names)
plt.tight_layout()
plt.savefig(results_model + '/confusionmatrix_test.png')

#will need this to compute loss term
df_cm = pd.DataFrame(confusion_matrix(labels_total,preds_total))
df_cm.to_csv(results_model + '/confusionmatrix_test.csv')

#classification report on test data

report =  classification_report(labels_total, preds_total, target_names = class_names, output_dict = True)
print(classification_report(labels_total, preds_total, target_names = class_names))

df = pd.DataFrame(report).transpose()
df.to_csv(results_model + '/classificationreport.csv')

##########################################################################################################################################################################  


if model_version.startswith("resnet"):
    target_layers = [model.layer4[-1]]
elif model_version.startswith("densenet"):
    target_layers = [model.features.denseblock4]
elif model_version.startswith("DINO"):    
    target_layers = [model.backbone.blocks[-1].norm1]
#target_layers = [model.layer4[-1]]

file_img = results_model + '/misclassified_images.pdf'

def save_misclassified_to_pdf(model, dataloader, class_names, target_layers,
                               device='cuda', output_file=file_img):
    model.eval()
    model.to(device)

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    inv_normalize = T.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )

    mis_loc_total = []
    mis_labels_total = []
    total_misclassified = 0

    with PdfPages(output_file) as pdf:
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            #if dataset_name == "cars":
            #    labels = labels.long().view(-1)
            
            if c == 1:
                images = images.repeat(1, 3, 1, 1).float()

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            misclassified = preds != labels
            mis_indices = misclassified.nonzero(as_tuple=True)[0]

            if len(mis_indices) == 0:
                continue  # No mistakes in this batch
                
            #if dataset_name == "sneakers":
            #    #files_to_use = test_filenames[mis_indices]
            #    files_to_use = [test_filenames[i] for i in mis_indices.tolist()]
            #    print(files_to_use[0])
            #    print(len(files_to_use))
                
            for idx in mis_indices:
                #print(idx)
                total_misclassified += 1
                img = images[idx]
                true = labels[idx].item()
                pred = preds[idx].item()
                prob = probs[idx]

                pred_conf = prob[pred].item() * 100
                true_conf = prob[true].item() * 100

                # Prepare for Grad-CAM
                input_tensor = img.unsqueeze(0).clone().detach().requires_grad_(True)
                targets = [ClassifierOutputTarget(pred)]

                # Inverse normalize for visualization
                rgb_img = inv_normalize(img.cpu()).permute(1, 2, 0).clamp(0, 1).numpy()

                # Grad-CAM
                with EigenCAM(model=model, target_layers=target_layers) as cam:
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                    grayscale_cam = grayscale_cam[0, :]
                    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                # Save to PDF
                fig, ax = plt.subplots(1, 2, figsize=(4, 4))
                
                ax[0].imshow(rgb_img)
                ax[0].axis('off')
                
                if dataset_name != "sneakers":
                    ax[0].set_title(
                        f"Missed Image\n"
                        f"True: {class_names[true]} ({true_conf:.1f}%)\n"
                        f"Pred: {class_names[pred]} ({pred_conf:.1f}%)",
                        fontsize=8
                    )
                elif dataset_name == "sneakers" or dataset_name == "cars":
                    #print(test_filenames[idx])
                    
                    dirname = os.path.basename(os.path.dirname(test_filenames[idx]))
                    filename = os.path.basename(test_filenames[idx])

                    new_var = os.path.join(dirname, filename)
                    
                    ax[0].set_title(
                        f"Missed Image\n"
                        f"Name: {new_var}\n"
                        f"True: {class_names[true]} ({true_conf:.1f}%)\n"
                        f"Pred: {class_names[pred]} ({pred_conf:.1f}%)",
                        fontsize=8
                    )

                ax[1].imshow(visualization)
                ax[1].axis('off')
                ax[1].set_title('GradCAM Results')
                
                #ax.imshow(visualization)
                #ax.axis('off')
                #ax.set_title(
                #    f"True: {class_names[true]} ({true_conf:.1f}%)\n"
                #    f"Pred: {class_names[pred]} ({pred_conf:.1f}%)",
                #    fontsize=8
                #)
                
                pdf.savefig(fig)
                plt.close(fig)

                mis_loc_total.append(idx.item())
                mis_labels_total.append(true)

        # If no misclassifications, add a note page
        if total_misclassified == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No misclassified images found", ha='center', va='center', fontsize=12)
            ax.axis('off')
            pdf.savefig(fig)
            plt.close(fig)

    # Save CSV
    df = pd.DataFrame({'Locations': mis_loc_total, 'Missed Image': mis_labels_total})
    df.to_csv(results_model + '/misclassified_images.csv', index=False)

    print(f"[INFO] Total misclassified: {total_misclassified}")
    print(f"[INFO] Saved PDF to {output_file}")
    print(f"[INFO] Saved CSV to misclassified_images.csv")

save_misclassified_to_pdf(model, test_loader, class_names, target_layers)

##########################################################################################################################################################################  

m = nn.Softmax(dim=1)
outputs_total = m(torch.Tensor(outputs_total))
outputs_total = outputs_total.cpu().detach().numpy()

#plt.style.use("seaborn")

plt.rc("font", size=12)
plt.rc("axes", labelsize=12)
plt.rc("xtick", labelsize=12)
plt.rc("ytick", labelsize=12)
plt.rc("legend", fontsize=12)

plt.rc("axes", titlesize=16)
plt.rc("figure", titlesize=16)
title = "Total Calibration Curve"

output_conf = np.max(outputs_total, axis=1)

fig = reliability_diagram(labels_total, preds_total, output_conf, num_bins=10, draw_ece=True,
                          draw_bin_importance="alpha", draw_averages=True,
                          title=title, figsize=(6, 6), dpi=100, 
                          return_fig=True)
fig.tight_layout()

fig.savefig(results_model + '/' + 'allclass_calibrationcurve' + '.pdf')
plt.close()

#other calibration scores

o = torch.Tensor(outputs_total)
l = torch.Tensor(labels_total)

metric1 = MulticlassCalibrationError(num_classes=len(class_names), n_bins=10, norm='l1')
ECE = metric1(o,l)
metric2 = MulticlassCalibrationError(num_classes=len(class_names), n_bins=10, norm='l2')
RMSCE = metric2(o,l)
metric3 = MulticlassCalibrationError(num_classes=len(class_names), n_bins=10, norm='max')
MCE = metric3(o,l)

print('ECE: %.4f' % (ECE))
print('RMSCE: %.4f' % (RMSCE))
print('MCE: %.4f' % (MCE))

#will need this to compute loss term
data = [['ECE', ECE], ['RMSCE', RMSCE], ['MCE', MCE]]
df_calmet = pd.DataFrame(data=data, columns=['Metric', 'Value'])
df_calmet.to_csv(results_model + '/calibrationmetrics.csv')
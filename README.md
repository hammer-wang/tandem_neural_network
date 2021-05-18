# Tandem neural network training for metasurface design

This is a PyTorch implementation of the work: https://onlinelibrary.wiley.com/doi/abs/10.1002/adma.201905467

Step 1: train the forward network  
Step 2: train the inverse network based on the pretrained forward network

Script for training:  
python train_forward_model.py --seed 42 --epochs 100 --device 0  
python train_inverse_model.py --seed 42 --epochs 100 --device 0
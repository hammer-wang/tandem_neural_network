# Tandem neural network training

Step 1: train the forward network  
Step 2: train the inverse network based on the pretrained forward network

Script for training:  
python train_forward_model.py --seed 42 --epochs 1000 --device 0  
python train_inverse_model.py --seed 42 --epochs 1000 --device 0
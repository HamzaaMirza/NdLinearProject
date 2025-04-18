#Hamza Imran Mirza
#Project for Ensemble AI Internship.

import torch
from ndlinear import NdLinear
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time

#Since, we already have imported the NdLinear module, we do not have to define it in the code. 

#The next step is that I will create a standard Projection Head using pytorch's nn.Linear layers
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        
        #Baseline projection head with two hidden layers (which can be changed later on)
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)
        

#Now, I will create a projection Head using NdLinear module. Which will replace the dense layers with NdLinear layers.
class NdProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):


        super(NdProjectionHead, self).__init__()
        self.net = nn.Sequential(

            #We will wrap the dimensions in a tuple, Hence we treat the feature vector as one dimension.
            NdLinear((in_dim,),(hidden_dim,), transform_outer=True),
            nn.ReLU(),
            NdLinear((hidden_dim,),(out_dim,), transform_outer=True) 

        )

    def forward(self, x):
        return self.net(x)
        

#The next step would be to create the SimCLR model.
#The model definition will contain the Encoder (encoder is our CNN) and the Projection Head (could be the baseline one or the one utilizing NdLinear layers.).
class SimCLR(nn.Module):

    def __init__(self, encoder, projection_head):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.projection_head = projection_head

    def forward(self, x):
        #the cnn encoder will extract the features from the input image
        features = self.encoder(x)

        #adding the ability to flatten the features
        features = features.view(features.size(0), -1)

        #Finally passing the features through the projeciton head that will produce the latent representations.
        projections = self.projection_head(features)
        return projections
    


#The next step would be to get the error function and what I will be using is NT-Xtent loss function as we are performing contrastive learning.
class NTXtentLoss(nn.Module):

    def __init__(self, temperature=0.5):
        super(NTXtentLoss, self).__init__()
        self.temperature = temperature


    def forward(self, z_i, z_j):
        # Normalizing each representation to unit vector
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        batch_size = z_i.size(0)


        #Concatenating the normalized embeddings
        representations = torch.cat([z_i, z_j], dim=0)

        #Calculating the similarity matrix
        sim_matrix = torch.matmul(representations, representations.T) / self.temperature

        #We need to nulilfy self similarities and for that we need to create a mask
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device =z_i.device)
        sim_matrix.masked_fill_(mask, -9e15)

        #Now we will calculate each sample's positive augmented counterpart.
        pos_indices = torch.arange(batch_size, 2 * batch_size, device = z_i.device)
        pos_indices = torch.cat([pos_indices, torch.arange(0, batch_size, device = z_i.device)])
        positives = sim_matrix[torch.arange(2 * batch_size, device= z_i.device), pos_indices]

        #Finally, we will compute the loss using log softmax formulation
        loss = -torch.log(torch.exp(positives) / torch.sum(torch.exp(sim_matrix), dim=1))
        return loss.mean()
    
#The next step would be to perform data augmentation for Self Supervised Learning.

class SimCLRAugmentation:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)


#Decided to add parameter counter inorder to better benchmark the models.
#The initial run used only Loss function to benchmark the models, but I thought it would be better to add the parameter counter as well.
#As then I will be able to see if NdLinear Layers uses lesser parameters than the standard nn.Linear layers.
#This will help me to see if NdLinear layers are more efficient than the standard nn.Linear layers.
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
        

if __name__ == '__main__':
    import os
    print("Current working directory:", os.getcwd())
    #The next step would be to create the dataset and the dataloader.

    #First I will set the device priorities
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    #The augmentations that will be performed on the input images are as follows:

    base_transform = transforms.Compose([
        transforms.RandomResizedCrop(96),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    #Wrapping the transform to generate two augmented views of the same image
    simclr_transform = SimCLRAugmentation(base_transform)


    #Next we will load the STL10 dataset
    train_dataset = datasets.STL10(root='./data', split='train', download=True, transform=simclr_transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)


    #The model that I will be using for this demonstration is ResNet18. 
    #This model will be used to create the encoder (CNN).

    resnet = models.resnet18(pretrained=False) #Since, we will be training the model from scratch, we will not use the pretrained weights.
    encoder = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten()) #Removing the last layer of the resnet model and flattening the output.

    #For ResNet18, the final feature dimensions are 512
    in_dim = 512
    hidden_dim = 256
    out_dim = 128

    #Now inorder to benchmark NdLinear's ability to use less parameters than the standard nn.Linear layers,
    # I will use reduced hidden dimenstions for Nd Projection Head
    reduced_hidden_dim = hidden_dim // 2  #For this example, I will be using 128 instead of 256

    #Now we will create two instances of the projection head, one with the baseline nn.Linear layers and the other with NdLinear layers.

    #The baseline head with standard dense layers is:
    baseline_head = ProjectionHead(in_dim, hidden_dim, out_dim)

    #The head using NdLinear layers is:
    nd_head = NdProjectionHead(in_dim, reduced_hidden_dim, out_dim)


    #Similarly, I will build two SimCLR models, each with one of the projection heads.
    #baseline mode is given as:
    baseline_model = SimCLR(encoder, baseline_head).to(device)

    #NdLinear layer model is given as:
    NdLinear_model = SimCLR(encoder, nd_head).to(device)



    #We also have to define the optimizer for both models.
    #for that we will use Adam optimizer with a learning rate of 1e-3.
    optimizer_baseline = optim.Adam(baseline_model.parameters(), lr=1e-3)
    optimizer_NdLinear = optim.Adam(NdLinear_model.parameters(), lr=1e-3)


    #We have already defined the loss function, so we will use that as well.
    #The loss function is NT-Xent loss function.
    criterion = NTXtentLoss(temperature=0.5)


    #Printing out the parameter count for both models.
    print("Baseline model parameters:", count_parameters(baseline_model))
    print("NdLinear model parameters:", count_parameters(NdLinear_model))

    # Lists to record losses and epoch times.
    losses_baseline = []
    losses_nd = []
    times_baseline = []
    times_nd = []

    #Finally, we can go ahead and define the taining loop
    def train_epoch(model, optimizer, data_loader, criterion, device):

        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Batches", ncols=100)
        for batch_idx, ((x1, x2), _) in progress_bar:
            x1 = x1.to(device)
            x2 = x2.to(device)
            optimizer.zero_grad()

            # Forward pass: get projections from both augmented views.
            z1 = model(x1)
            z2 = model(x2)

            # Compute the contrastive loss.
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()
        
            epoch_loss += loss.item()
            # Update the progress bar with the current batch loss.
            progress_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(data_loader)
        return avg_loss

    # Main training loop that prints epoch information.
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Time the baseline model
        start_time = time.time()
        loss_baseline = train_epoch(baseline_model, optimizer_baseline, train_loader, criterion, device)
        baseline_epoch_time = time.time() - start_time
        
        # Time the NdLinear model
        start_time = time.time()
        loss_nd = train_epoch(NdLinear_model, optimizer_NdLinear, train_loader, criterion, device)
        ndlinear_epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch} - Baseline Loss: {loss_baseline:.4f} (Time: {baseline_epoch_time:.2f}s), "
              f"NdLinear Loss: {loss_nd:.4f} (Time: {ndlinear_epoch_time:.2f}s)")
        

        losses_baseline.append(loss_baseline)
        losses_nd.append(loss_nd)
        times_baseline.append(baseline_epoch_time)
        times_nd.append(ndlinear_epoch_time)
        

    #Now we will Compute the epoch at which each model reaches a given convergence threshold.
    convergence_threshold = 3.5  # Set an example threshold for NT-Xent loss.
    baseline_convergence_epoch = None
    nd_convergence_epoch = None
    for i in range(len(losses_baseline)):
        if baseline_convergence_epoch is None and losses_baseline[i] < convergence_threshold:
            baseline_convergence_epoch = i + 1
        if nd_convergence_epoch is None and losses_nd[i] < convergence_threshold:
            nd_convergence_epoch = i + 1
    
    print("\nConvergence Summary:")
    print("Baseline model converged at epoch:", baseline_convergence_epoch if baseline_convergence_epoch is not None else "Not reached")
    print("NdLinear model converged at epoch:", nd_convergence_epoch if nd_convergence_epoch is not None else "Not reached")
    


    #Now we will plot the graphs:
    import matplotlib.pyplot as plt

    #Plotting the training loss curves for both models
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), losses_baseline, label="Baseline Model")
    plt.plot(range(1,  num_epochs + 1), losses_nd, label="NdLinear Model")
    plt.xlabel("Epoch")
    plt.ylabel("Average NT-Xent Loss")
    plt.title("Convergence Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()


    #Plotting the epoch times for both modekls.
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), times_baseline, label="Baseline Time per Epoch")
    plt.plot(range(1, num_epochs + 1), times_nd, label="NdLinear Time per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Time (s)")
    plt.title("Epoch Training Time Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()
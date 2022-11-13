#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data.distributed#

import argparse
import logging#
import json
import sys
import os
from PIL import ImageFile

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader,criterion,device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    running_loss=0
    running_corrects=0
    
    with torch.no_grad():#disables gradients, good for inference
        for inputs, labels in test_loader:
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs=model(inputs)
            loss=criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects/ len(test_loader.dataset)
    logger.info(f"Testing Accuracy: {100*total_acc}, Testing Loss: {total_loss}")
    print(f"Testing Accuracy: {100*total_acc}, Testing Loss: {total_loss}")

def train(model, train_loader, valid_loader, criterion, optimizer, device,epochs):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':valid_loader}
    loss_counter=0
    
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_samples=0
            
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples+=len(inputs)
                if running_samples % 2000  == 0:
                    accuracy = running_corrects/running_samples
                    logger.info("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                            running_samples,
                            len(image_dataset[phase].dataset),
                            100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0*accuracy,
                        )
                    )
                    print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                            running_samples,
                            len(image_dataset[phase].dataset),
                            100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0*accuracy,
                        )
                    )
                
                #NOTE: Comment lines below to train and test on whole dataset
                if running_samples>(0.2*len(image_dataset[phase].dataset)):
                    break

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples
            accuracy=100*epoch_acc
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                    logger.info(f"For validation, Epoch: {epoch}, best loss{best_loss}")
                else:
                    loss_counter+=1
                
            logger.info(f"Epoch: {epoch}, Loss: {epoch_loss}, Accuracy: {accuracy}%")
        if loss_counter==1:
            break
    return model

    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False # Means freeze convolutional layer
    num_features = model.fc.in_features #number of features
    model.fc=nn.Sequential(
            nn.Linear(num_features,133)
    )
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    logger.info("Get train data loader")
    train_transformation = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),

        #Resize the input image to the given size. 255
        transforms.Resize(255),

        #Crops the given image at the center.
        transforms.CenterCrop(224),
        
        #Randomly convert image to grayscale with a probability of p (default 0.1).
        transforms.RandomGrayscale(p=0.3),

        # transform to tensors
        transforms.ToTensor(),

        # Normalize the pixel values (in R, G, and B channels)
        #Normalize a tensor image with mean and standard deviation.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        

    logger.info("Get validation data loader")
    valid_transformation=transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.RandomGrayscale(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    logger.info("Get test data loader")
    test_transformation=transforms.Compose([
        #transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(255),
        transforms.CenterCrop(224),
        #transforms.RandomGrayscale(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    #get the path to the data.
    train_dir=os.path.join(data,'train')
    valid_dir=os.path.join(data,'valid')
    test_dir = os.path.join(data,'test')

    # Load all of the images, transforming them
    train_dataset = torchvision.datasets.ImageFolder(
        root=train_dir,
        transform=train_transformation
        )
    test_dataset = torchvision.datasets.ImageFolder(
        root=test_dir,
        transform=test_transformation
        )
    valid_dataset=torchvision.datasets.ImageFolder(
        root=valid_dir,
        transform=valid_transformation
        )


    

    # define a loader for the training data we can iterate through in n-image batches
    #n can be 64 as batch_size
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
        )
    
    # define a loader for the testing data we can iterate through in n-image batches
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
        )
    valid_loader =torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False
        )
    return train_loader, test_loader,valid_loader

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    torch.save(model.cpu().state_dict(), path)
def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=net()
    model=model.to(device)
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_loader,test_loader,valid_loader=create_data_loaders(args.data_dir, args.batch_size)
    #model=train(model, train_loader, loss_criterion, optimizer)
    model=train(model, train_loader, valid_loader,loss_criterion, optimizer,device,args.epochs)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion,device)
    
    '''
    TODO: Save the trained model
    '''
    logger.info("Saving the model.")
    save_model(model, args.model_dir)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
        
  
    #print("Default Bucket: {}".format(bucket))
    # Container environment
    #parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
   # parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    #os.environ['SM_OUTPUT_DATA_DIR']=f's3://{bucket}/dogImages/output/' #location of other artifacts
   # parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
        
    args=parser.parse_args()
   
    main(args)

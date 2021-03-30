import torch
import numpy as np
from tqdm import tqdm
from utils import save_checkpoint


def evaluate(dataset, data_loader, model, epoch, criterion, device):

    # put model in eval mode
    model.eval()
    # init final_loss to 0
    final_loss = 0
    # calculate number of batches and init tqdm
    num_batches = int(len(dataset) / data_loader.batch_size)
    tk0 = tqdm(data_loader, total=num_batches)
    # we need no_grad context of torch. this save memory
    with torch.no_grad():
        for inputs, targets in tk0:
            tk0.set_description(f"Validation Epoch {epoch+1}")
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.long)
            output = model(inputs)
            loss = criterion(output, targets)
            # add loss to final loss
            final_loss += loss
            tk0.set_postfix(loss = (final_loss / num_batches).item())
    # close tqdm
    tk0.close()
    # return average loss over all batches
    return final_loss / num_batches


def train(traindataset, validdataset, trainloader, validloader, model, criterion, optimizer, epochs, checkpoint_name, device):

    if torch.cuda.is_available():
        criterion = criterion.cuda()

    # calculate number of batches
    num_batches = int(len(traindataset)/trainloader.batch_size)
    
    train_loss = []
    valid_loss = []
    model_weights = []
    valid_loss_min = np.Inf
    
    for epoch in range (epochs):
        # putting the model in train mode
        model.train()

        with tqdm(trainloader, unit="batch") as tk0:
            for inputs, targets in tk0:
                tk0.set_description(f"Training Epoch {epoch+1}")
                # fetch input images and masks from dataset batch
                # move images and masks to cpu/gpu device
                inputs = inputs.to(device, dtype=torch.float)
                targets = targets.to(device, dtype=torch.long)
                # zero grad the optimizer
                optimizer.zero_grad()

                # forward step of model
                outputs = model(inputs)

                # calculate loss
                loss = criterion(outputs, targets)

                loss.backward()
                # step the optimizer
                optimizer.step()
                tk0.set_postfix(loss=loss.item())
                        
        val_loss_log = evaluate(
            validdataset,
            validloader,
            model,
            epoch,
            criterion,
            device
        )
        
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        
        if epoch % 1 == 0:
            save_checkpoint(checkpoint, filename = checkpoint_name + str(epoch) + ".pth.tar")
            
        train_loss.append(loss.item())
        valid_loss.append(val_loss_log.item())
      
    return train_loss, model_weights, valid_loss

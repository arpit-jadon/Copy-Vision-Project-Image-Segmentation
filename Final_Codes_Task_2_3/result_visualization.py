import torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from utils import load_checkpoint
from metrics import eval_metrics
from loader import cityscapesLoader
import matplotlib.pyplot as plt


def result_vis(checkpoint_path, dataset, loader, bs, model, device):
    load_checkpoint(checkpoint_path, model)

    model.eval()

    with torch.no_grad():
        for batch_idx, (images,labels) in enumerate(loader):
            # Number of batches to analyze
            if batch_idx>0:
                break
            #output = model(images.cuda())
            images, labels = images.to(device), labels.to(device)

            output = torch.sigmoid(model(images))
            output_2d = torch.argmax(output, dim=1).detach().cpu().numpy()

            imgs = images.cpu().numpy().transpose([0,2,3,1])

            for img_num in range(0,bs): 
                plt.figure(figsize=(15,20))

                plt.subplot(1,3,1).set_title('Original_Image')
                plt.imshow(np.clip(imgs[img_num,:,:,:], 0, 1))

                plt.subplot(1,3,2).set_title('Ground Truth')
                GT_mask_rgb = dataset.decode_segmap(labels.cpu().numpy()[img_num])
                plt.imshow(GT_mask_rgb) # Single channel GT mask to RGB Image

                plt.subplot(1,3,3).set_title('Prediction_UNet')
                pred_mask_rgb = dataset.decode_segmap(output_2d[img_num])
                plt.imshow(pred_mask_rgb) # Single channel prediction mask to RGB Image

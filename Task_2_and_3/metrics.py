import torch
import numpy as np
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")

# +
EPS = 1e-10


def nanmean(x):
    return torch.mean(x[x == x])


def conf_matrix(true, pred, num_classes):
    mask = (true >= 0) & (true < num_classes)
    conf_mat = torch.bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float()
    return conf_mat


def iou(conf_mat):
    A_inter_B = torch.diag(conf_mat)
    A = conf_mat.sum(dim=1)
    B = conf_mat.sum(dim=0)
    jaccard = A_inter_B / (A + B - A_inter_B + EPS)
    avg_jacc = nanmean(jaccard)
    return avg_jacc

def per_class_pixel_accuracy(conf_mat):
    correct_per_class = torch.diag(conf_mat)
    total_per_class = conf_mat.sum(dim=1)
    per_class_acc = correct_per_class / (total_per_class + EPS)
    avg_per_class_acc = nanmean(per_class_acc)
    return avg_per_class_acc


def dice_score(conf_mat):
    A_inter_B = torch.diag(conf_mat)
    A = conf_mat.sum(dim=1)
    B = conf_mat.sum(dim=0)
    dice = (2 * A_inter_B) / (A + B + EPS)
    avg_dice = nanmean(dice)
    return avg_dice

def roc_auc(actual_class, pred_class, average = "macro"):
    unique_class = np.unique(actual_class)
    roc_auc_img = 0
    
    new_actual_class = np.zeros_like(actual_class)
    new_pred_class = np.zeros_like(pred_class)
    
    for per_class in unique_class:
        
        new_actual_class[actual_class == per_class] = 1
        new_pred_class[pred_class == per_class] = 1

        try:
            roc_auc = metrics.roc_auc_score(new_actual_class.flatten(), new_pred_class.flatten(), average = average)
        except ValueError:
            pass
        
        roc_auc_img += roc_auc

    return roc_auc_img/len(unique_class)

def eval_metrics(true, pred, num_classes, batch_sz):
    conf_mat = torch.zeros((num_classes, num_classes))
    f1_score = 0
    auc_score = 0
    # true--> 2d batch labels (8,512,512) and 2d batch pred---> (8,512,512) 
    for t, p in zip(true, pred):
        # flattening 1 label channel or 1 mask at a time
        conf_mat += conf_matrix(t.flatten(), p.flatten(), num_classes)
        f1_score += metrics.f1_score(t.flatten().numpy(), p.flatten().numpy(), average='macro')
        auc_score += roc_auc(t.numpy(), p.numpy())
        
    accuracy_per_class = per_class_pixel_accuracy(conf_mat)
    avg_iou = iou(conf_mat)
    avg_dice = dice_score(conf_mat)
    return avg_iou, avg_dice, f1_score/batch_sz, accuracy_per_class, auc_score/batch_sz
# -



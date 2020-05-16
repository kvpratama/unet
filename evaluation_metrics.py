import torch


# prediction : Segmentation Result
# gt : Ground Truth

def accuracy(prediction, gt, threshold=0.5):
    prediction = prediction > threshold
    gt = gt == torch.max(gt)
    corr = torch.sum(prediction == gt)
    tensor_size = prediction.size(0) * prediction.size(1) * prediction.size(2) * prediction.size(3)
    acc = float(corr) / float(tensor_size)

    return acc


def sensitivity(prediction, gt, threshold=0.5):
    # Sensitivity == Recall
    prediction = prediction > threshold
    gt = gt == torch.max(gt)

    # TP : True Positive
    # FN : False Negative
    TP = ((prediction == 1) + (gt == 1)) == 2
    FN = ((prediction == 0) + (gt == 1)) == 2

    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

    return SE


def specificity(prediction, gt, threshold=0.5):
    prediction = prediction > threshold
    gt = gt == torch.max(gt)

    # TN : True Negative
    # FP : False Positive
    TN = ((prediction == 0) + (gt == 0)) == 2
    FP = ((prediction == 1) + (gt == 0)) == 2

    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

    return SP


def precision(prediction, gt, threshold=0.5):
    prediction = prediction > threshold
    gt = gt == torch.max(gt)

    # TP : True Positive
    # FP : False Positive
    TP = ((prediction == 1) + (gt == 1)) == 2
    FP = ((prediction == 1) + (gt == 0)) == 2

    PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)

    return PC


def F1(prediction, gt, threshold=0.5):
    # Sensitivity == Recall
    SE = sensitivity(prediction, gt, threshold=threshold)
    PC = precision(prediction, gt, threshold=threshold)

    F1 = 2 * SE * PC / (SE + PC + 1e-6)

    return F1


def jaccard(prediction, gt, threshold=0.5):
    # JS : Jaccard similarity
    prediction = prediction > threshold
    gt = gt == torch.max(gt)

    Inter = torch.sum((prediction + gt) == 2)
    Union = torch.sum((prediction + gt) >= 1)

    JS = float(Inter) / (float(Union) + 1e-6)

    return JS


def dice(prediction, gt, threshold=0.5):
    # DC : Dice Coefficient
    prediction = prediction > threshold
    gt = gt == torch.max(gt)

    Inter = torch.sum((prediction + gt) == 2)
    DC = float(2 * Inter) / (float(torch.sum(prediction) + torch.sum(gt)) + 1e-6)

    return DC
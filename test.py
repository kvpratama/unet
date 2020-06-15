import os
import argparse
import glob
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

import load_model
import load_dataset
import evaluation_metrics

import matplotlib.pyplot as plt
import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="./saved_models/", help="checkpoint directory")
    opt = parser.parse_args()
    print(opt)

    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    table = []
    checkpoints = [f for f in glob.glob(os.path.join(opt.checkpoint_dir, '*.pth'))]
    for checkpoint in checkpoints:
        print("Starting Checkpoint", checkpoint)
        state = torch.load(checkpoint)
        saved_opt = state.get('opt')
        model = load_model.load_model(saved_opt)
        model.load_state_dict(state.get('weight', False))
        if cuda:
            model.cuda()

        saved_opt.test = True

        dataloader = DataLoader(
            load_dataset.get_dataset(saved_opt),
            batch_size=saved_opt.batch_size,
            shuffle=False,
            num_workers=saved_opt.n_cpu,
        )

        for i, imgs in enumerate(dataloader):
            row = []
            # Configure model input
            data = imgs["input"].type(Tensor)
            true_mask = imgs["gt"].type(Tensor)

            model.eval()
            with torch.no_grad():
                predicted_mask = model(data)
                if saved_opt.deep_supervision:
                    mean = torch.mean(torch.stack(predicted_mask), dim=0)
                    predicted_mask = torch.sigmoid(mean)
                elif saved_opt.n_class > 1:
                    predicted_mask = F.softmax(predicted_mask, dim=1)
                else:
                    predicted_mask = torch.sigmoid(predicted_mask)

            row.append(imgs["filepath"][0])
            row.append(checkpoint)
            row.append(evaluation_metrics.accuracy(predicted_mask, true_mask))
            row.append(evaluation_metrics.sensitivity(predicted_mask, true_mask))
            row.append(evaluation_metrics.specificity(predicted_mask, true_mask))
            row.append(evaluation_metrics.precision(predicted_mask, true_mask))
            row.append(evaluation_metrics.F1(predicted_mask, true_mask))
            row.append(evaluation_metrics.jaccard(predicted_mask, true_mask))
            row.append(evaluation_metrics.dice(predicted_mask, true_mask))

            table.append(row)
    table_df = pd.DataFrame(table, columns=['Filepath', 'Checkpoint', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1', 'Jaccard', 'Dice'])
    table_df['Mean'] = table_df[['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1', 'Jaccard', 'Dice']].mean(axis=1)
    table_df.groupby('Checkpoint').mean().to_csv(os.path.join(opt.checkpoint_dir, 'evaluation.csv'))

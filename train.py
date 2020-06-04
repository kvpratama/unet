import argparse
import os

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

import load_model
import load_dataset

import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="carvana", help="dataset name")
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--model_name", type=str, default="unet", help="model name")
    parser.add_argument('--scale', dest='scale', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument("--in_channels", type=int, default=3, help="number of input channels")
    parser.add_argument("--n_class", type=int, default=1, help="number of class")
    parser.add_argument("--checkpoint_dir", type=str, default="./saved_models/", help="checkpoint directory")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
    parser.add_argument("--test", action='store_true', help="Run model on test set")
    opt = parser.parse_args()

    os.makedirs("%s/" % opt.checkpoint_dir, exist_ok=True)

    opt_dict = vars(opt)
    with open(os.path.join(opt.checkpoint_dir, 'opt.csv'), 'w') as f:
        for key in opt_dict.keys():
            f.write("%s,%s\n" % (key, opt_dict[key]))

    print(opt)

    cuda = torch.cuda.is_available()

    model = load_model.load_model(opt)

    if cuda:
        model.cuda()

    if opt.epoch != 0:
        # Load pretrained models
        temp_opt = opt
        state = torch.load("%s/%d.pth" % (opt.checkpoint_dir, opt.epoch))
        model.load_state_dict(state.get('weight', False))
        opt = state.get('opt')
        opt.epoch = temp_opt.epoch

    if opt.n_class > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters())

    dataloader = DataLoader(
        load_dataset.get_dataset(opt),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    log_list = []
    for epoch in range(opt.epoch, opt.n_epochs):
        epoch_loss = 0
        num_batches = len(dataloader)

        for i, imgs in enumerate(dataloader):
            # Configure model input
            data = Variable(imgs["input"].type(Tensor))
            true_mask = Variable(imgs["gt"].type(Tensor))

            predict_mask = model(data)

            loss = criterion(predict_mask, true_mask)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('ep: [{0:.0f}/{1:.0f}] batch: [{2:.0f}/{3:.0f}] loss: {4:.6f}'.format(epoch + 1, opt.n_epochs, i + 1,
                                                                                        num_batches, loss.item()))
        log_list.append(epoch_loss)
        log_df = pd.DataFrame(log_list, columns=['loss'])
        log_df.to_csv(os.path.join(opt.checkpoint_dir, 'loss.csv'))

        if epoch % opt.checkpoint_interval == 0:
            try:
                weight = model.module.state_dict()
            except:
                weight = model.state_dict()

            state = {'opt': opt, 'weight': weight}
            torch.save(state, "%s/%d.pth" % (opt.checkpoint_dir, epoch+1))

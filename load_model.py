from model import unet_model


def load_model(opt):

    if opt.model_name == 'unet':
        model = unet_model.UNet(n_channels=opt.n_channels, n_classes=opt.n_class)
    else:
        model = unet_model.UNet(n_channels=opt.n_channels, n_classes=opt.n_class)

    return model

from model import unet_model, unet_attention


def load_model(opt):

    if opt.model_name == 'unet':
        model = unet_model.UNet(in_channels=opt.in_channels, n_classes=opt.n_class)
    elif opt.model_name == 'unet_attention':
        model = unet_attention.UNetAttention(in_channels=opt.in_channels, n_classes=opt.n_class)
    else:
        model = unet_model.UNet(in_channels=opt.in_channels, n_classes=opt.n_class)

    return model

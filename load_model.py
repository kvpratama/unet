from model import unet_model, unet_attention, unet_recurrent


def load_model(opt):

    if opt.model_name == 'unet':
        model = unet_model.UNet(in_channels=opt.in_channels, n_classes=opt.n_class)
    elif opt.model_name == 'unet_attention':
        model = unet_attention.UNetAttention(in_channels=opt.in_channels, n_classes=opt.n_class)
    elif opt.model_name == 'unet_recurrent':
        model = unet_recurrent.RecurrentUNet(in_channels=opt.in_channels, n_classes=opt.n_class)
    else:
        model = unet_model.UNet(in_channels=opt.in_channels, n_classes=opt.n_class)

    return model

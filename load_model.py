from model import unet_model, unet_attention, unet_recurrent, R2UNet, unet_nested


def load_model(opt):

    if opt.model_name == 'unet':
        model = unet_model.UNet(in_channels=opt.in_channels, n_classes=opt.n_class)
    elif opt.model_name == 'unet_attention':
        model = unet_attention.UNetAttention(in_channels=opt.in_channels, n_classes=opt.n_class)
    elif opt.model_name == 'unet_recurrent':
        model = unet_recurrent.RecurrentUNet(in_channels=opt.in_channels, n_classes=opt.n_class)
    elif opt.model_name == 'unet_r2':
        model = R2UNet.R2UNet(in_channels=opt.in_channels, n_classes=opt.n_class)
    elif opt.model_name == 'unet_nested':
        model = unet_nested.NestedUNet(opt)
    else:
        print("WARNING! Model not found. Using standard UNet model")
        model = unet_model.UNet(in_channels=opt.in_channels, n_classes=opt.n_class)

    return model

from dataset import carvana_dataset, isic_dataset, dsb18_dataset


def get_dataset(opt):
    if opt.dataset_name == 'carvana':
        return carvana_dataset.CarvanaDataset(opt)
    elif opt.dataset_name == 'isic':
        return isic_dataset.IsicDataset(opt)
    elif opt.dataset_name == 'dsb18':
        return dsb18_dataset.DSB18Dataset(opt)
    else:
        print('Dataset is not found. Loading Carvana dataset')
        return carvana_dataset.CarvanaDataset(opt)

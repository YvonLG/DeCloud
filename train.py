



if __name__ == '__main__':
    from options import get_parser
    from models.model import DeCloudGAN
    from models.utils import seed_everything
    import torch
    from torch.utils.data import DataLoader

    from datasets.datasets import TimeSerieDS

    from tqdm import tqdm

    opt = get_parser().parse_args()
    opt.output_nc = 3
    opt.input_nc = 2*2+3*1
    opt.weights_init = 'normal'
    opt.n_encode = 0
    opt.beta1 = 0.5
    opt.num_downs = 8
    opt.lr = 0.0002
    opt.n_critics = 5
    opt.gan_mode = 'wgangp'
    opt.upsample = 'bilinear'
    
    ds_train = TimeSerieDS.from_save('train_dataset.pickle', ts_len=2, s2_ch=[3,2,1])
    ds_valid = TimeSerieDS.from_save('valid_dataset.pickle', ts_len=2, s2_ch=[3,2,1])
    dl_train = DataLoader(ds_train, batch_size=10, shuffle=True)
    dl_valid = DataLoader(ds_valid, batch_size=10)

    seed_everything(42)

    gan = DeCloudGAN(opt)

    for i in range(20):
        gan.train_dataloader(dl_valid)
        gan.valid_dataloader(dl_valid)




    
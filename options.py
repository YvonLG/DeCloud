
import argparse

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()


    # general
    parser.add_argument('--device', type=str, default='cuda', help='cuda|cpu')
    parser.add_argument('--seed', type=int, help='random seed')

    # adam optimizer
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta 1 parameter')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta 2 parameter')

    # generator + discriminator
    parser.add_argument('--input_nc', type=int, default=2, help='# of input channel')
    parser.add_argument('--output_nc', type=int, default=1, help='# of output channel')
    parser.add_argument('--n_encode', type=int, default=5, help='# of encoding features')
    parser.add_argument('--norm', type=str, default='instance', help='instance|batch|none normalization')
    parser.add_argument('--weights_init', type=str, default='normal', help='normal|xavier|kaiming|orthogonal weights initialization')

    # generator
    parser.add_argument('--ngf', type=int, default=64, help='# channel in gen first conv')
    parser.add_argument('--num_downs', type=int, default=8, help='# of downsampling, make sure 2 ** num_downs <= input width')
    parser.add_argument('--upsample', type=str, default='basic', help='basic|bilinear conv transpose or upsample -> bilinear interp -> conv')
    parser.add_argument('--dropout', type=bool, default=True, action=argparse.BooleanOptionalAction, help='use dropouts as presented in the original pix2pix paper')
    parser.add_argument('--g_where_add', type=str, default='all', help='all|input where to concat encoding features')

    # discriminator
    parser.add_argument('--ndf', type=int, default=64, help='# channel in disrc first conv')
    parser.add_argument('--d_where_add', type=str, default='input', help='see --g_where_add')

    # gan training parameters
    parser.add_argument('--gan_mode', type=str, default='vanilla', help='vanilla|lsgan|wgangp')
    parser.add_argument('--lambda_l1', type=float, default=100, help='coef of l1 loss')
    parser.add_argument('--lambda_ssim', type=float, default=0, help='coef of ssim loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='coef of gradient penalty')
    parser.add_argument('--n_critics', type=int, default=5, help='# number of critic in wgpangp')

    # datasets
    parser.add_argument('--train_dataset', type=str, default='train_dataset.pickle', help='path to the train dataset saved state')
    parser.add_argument('--valid_dataset', type=str, default='valid_dataset.pickle', help='path to the valid dataset saved state')
    
    return parser

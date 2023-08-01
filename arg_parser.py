import argparse


def get_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--L', type=int, default=128,
                        help='training sequence length')
    parser.add_argument('--filename', type=str, default='test.pkl',
                        help='dataset filename. file is expected in ./data/')
    parser.add_argument('--frac-train', type=float, default=0.8,
                        help='training sequence length')
    parser.add_argument('--frac-valid', type=float, default=0.25,
                        help='training sequence length')
    parser.add_argument('--train-batch', type=int, default=32,
                        help='training sequence length')
    parser.add_argument('--varlen_train', action='store_true', default=False,
                        help='enable variable length training')
    parser.add_argument('--use-error', action='store_true', default=False,
                        help='use error as additional dimension')
    parser.add_argument('--use-meta', action='store_true', default=False,
                        help='use meta as auxiliary network input')
    parser.add_argument('--input', type=str, default='dtf',
                        help='obsolete. input representation of data. use either dtf or dtfe, which include errors')
    parser.add_argument('--n_test', type=int, default=1,
                        help='number of different sequence length to test')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--dropout', type=float, default=0,
                        help='dropout rate')
    parser.add_argument('--dropout-classifier', type=float, default=0,
                        help='dropout rate')
    parser.add_argument('--permute', action='store_true', default=False,
                        help='data augmentation')
    parser.add_argument('--clip', type=float, default=-1,
                        help='gradient clipping')
    parser.add_argument('--path', type=str, default='temp',
                        help='folder name to save experiement results')
    parser.add_argument('--max_epoch', type=int, default=50,
                        help='maximum number of training epochs')
    parser.add_argument('--min_maxpool', type=int, default=2,
                        help='minimum length required for maxpool operation.')
    parser.add_argument('--ngpu', type=int, default=1,
                        help='number of gpu devices to use. neg number refer to particular single device number')
    parser.add_argument('--njob', type=int, default=1,
                        help='maximum number of networks to train on each gpu')
    parser.add_argument('--K', type=int, default=1,
                        help='number of data partition to use')
    parser.add_argument('--pseed', type=int, default=0,
                        help='random seed for data partition (only when K = 1)')
    parser.add_argument('--network', type=str, default='iresnet',
                        help='name of the neural network to train')
    parser.add_argument('--kernel', type=int, default=2,
                        help='kernel size')
    parser.add_argument('--depth', type=int, default=7,
                        help='network depth')
    parser.add_argument('--n_layer', type=int, default=2,
                        help='(iresnet/resnet only) number of convolution per residual block')
    parser.add_argument('--hidden', type=int, default=128,
                        help='hidden dimension')
    parser.add_argument('--hidden-classifier', type=int, default=32,
                        help='hidden dimension for final layer')
    parser.add_argument('--max_hidden', type=int, default=128,
                        help='(iresnet/resnet only) maximum hidden dimension')
    parser.add_argument('--two_phase', action='store_true', default=False,
                        help='')
    parser.add_argument('--print_every', type=int, default=-1,
                        help='')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for network seed and random partition')
    parser.add_argument('--cudnn_deterministic', action='store_true', default=False,
                        help='')
    parser.add_argument('--min_sample', type=int, default=0,
                        help='minimum number of pre-segmented light curve per class')
    parser.add_argument('--max_sample', type=int, default=100000,
                        help='maximum number of pre-segmented light curve per class during testing')
    parser.add_argument('--test', action='store_true', default=False,
                        help='test pre-trained models')
    parser.add_argument('--retrain', action='store_true', default=False,
                        help='continue training from checkpoint')
    parser.add_argument('--no-log', action='store_true', default=False,
                        help='continue training from checkpoint')
    parser.add_argument('--note', type=str, default='',
                        help='')
    parser.add_argument('--project-name', type=str, default='',
                        help='for weights and biases tracking')
    parser.add_argument('--decay-type', type=str, default='plateau',
                        help='')
    parser.add_argument('--patience', type=int, default=5,
                        help='patience for learning decay')
    parser.add_argument('--early_stopping', type=int, default=0,
                        help='terminate training if loss does not improve by 10% after waiting this number of epochs')

    args = parser.parse_args()

    return args

import argparse


def get_args(notebook=False):

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--dataset', type=str, default='macho',
                        help='dataset filename. data is expected in ./data/')
    parser.add_argument('--path', type=str, default='results',
                        help='folder name to save experiment results')

    parser.add_argument('--max_epoch', type=int, default=50,
                        help='maximum number of training epochs')
    parser.add_argument('--train-batch', type=int, default=32,
                        help='training sequence length')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')

    parser.add_argument('--network', type=str, default='lstm',
                        help='name of the neural network to train')
    parser.add_argument('--dropout', type=float, default=0.15,
                        help='dropout rate for all layers except for the final classifier')
    parser.add_argument('--dropout-classifier', type=float, default=0,
                        help='dropout rate for the final classifier')
    parser.add_argument('--min_maxpool', type=int, default=2,
                        help='minimum length required for maxpool operation.')
    parser.add_argument('--kernel', type=int, default=2,
                        help='kernel size')
    parser.add_argument('--depth', type=int, default=2,
                        help='network depth')
    parser.add_argument('--n_layer', type=int, default=2,
                        help='(iresnet/resnet only) number of convolution per residual block')
    parser.add_argument('--hidden', type=int, default=12,
                        help='hidden dimension')
    parser.add_argument('--hidden-classifier', type=int, default=32,
                        help='hidden dimension for final layer')
    parser.add_argument('--max_hidden', type=int, default=128,
                        help='(iresnet/resnet only) maximum hidden dimension')

    parser.add_argument('--permute', action='store_true', default=False,
                        help='data augmentation')
    parser.add_argument('--clip', type=float, default=-1,
                        help='gradient clipping')

    parser.add_argument('--print_every', type=int, default=-1,
                        help='')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for network seed and random partition')
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

    if notebook:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    return args

from models.iresnet import IResNet
from models.itcn import ITCN
from models.rnn import RNN


def get_network(n_inputs, n_classes, args, dtype):

    if args.network in ['itcn', 'iresnet']:
        padding = 'cyclic'
    else:
        padding = 'zero'

    if args.network in ['itcn', 'tcn']:
        clf = ITCN(
            num_inputs=n_inputs,
            num_class=n_classes,
            depth=args.depth,
            hidden_conv=args.hidden,
            hidden_classifier=args.hidden_classifier,
            dropout=args.dropout,
            kernel_size=args.kernel,
            dropout_classifier=args.dropout_classifier,
            aux=3,
            padding=padding
        ).type(dtype)

    elif args.network in ['iresnet', 'resnet']:
        clf = IResNet(
            n_inputs,
            n_classes,
            depth=args.depth,
            nlayer=args.n_layer,
            kernel_size=args.kernel,
            hidden_conv=args.hidden,
            max_hidden=args.max_hidden,
            padding=padding,
            min_length=args.min_maxpool,
            aux=3,
            dropout_classifier=args.dropout_classifier,
            hidden=args.hidden_classifier
        ).type(dtype)

    elif args.network in ['gru', 'lstm']:
        clf = RNN(
            num_inputs=n_inputs,
            hidden_rnn=args.hidden,
            num_layers=args.depth,
            num_class=n_classes,
            hidden=args.hidden_classifier,
            rnn=args.network.upper(),
            dropout=args.dropout,
            aux=3
        ).type(dtype)

    else:
        raise NotImplementedError(f'{args.network} is not implemented')

    return clf

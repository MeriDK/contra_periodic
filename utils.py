from models.iresnet import IResNet
from models.itcn import ITCN
from models.rnn import RNN


def get_model(n_inputs, n_classes, args, dtype):

    if args.network in ['itcn', 'iresnet']:
        padding = 'cyclic'
    else:
        padding = 'zero'

    if args.network in ['itcn', 'tcn']:
        model = ITCN(
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
        model = IResNet(
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
        model = RNN(
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

    return model


def create_save_name(args):
    # Create a base format for the filename
    base_format = "{dataset}-{network}-K{kernel}-D{depth}-H{hidden}-LR{lr}-CLIP{clip}-DROP{dropout}"

    # Create a dictionary of arguments
    args_dict = vars(args).copy()

    # Update the dictionary and format string based on the network type
    if args.network in ['resnet', 'iresnet']:
        base_format = base_format.replace("-CLIP{clip}", "-NL{n_layer}-MH{max_hidden}-CLIP{hidden_classifier}")
        args_dict["dropout"] = max(args.dropout, args.dropout_classifier)

    # Use the format string and dictionary to create the save name
    save_name = base_format.format(**args_dict)

    return save_name

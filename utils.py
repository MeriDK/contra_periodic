import os
import shutil


def create_save_name(args):
    # Create a base format for the filename
    base_format = "{filename}-{network}-K{kernel}-D{depth}-H{hidden}-L{L}-V{varlen_train}-{input}-LR{lr}-CLIP{clip}-" \
                  "DROP{dropout}-TP{two_phase}"

    # Create a dictionary of arguments
    args_dict = vars(args).copy()
    args_dict["filename"] = args.filename[:-4]
    args_dict["varlen_train"] = int(args.varlen_train)
    args_dict["two_phase"] = int(args.two_phase)

    # Update the dictionary and format string based on the network type
    if args.network in ['resnet', 'iresnet']:
        base_format = base_format.replace("-CLIP{clip}", "-NL{n_layer}-MH{max_hidden}-CLIP{hidden_classifier}")
        args_dict["dropout"] = max(args.dropout, args.dropout_classifier)

    # Use the format string and dictionary to create the save name
    save_name = base_format.format(**args_dict)

    return save_name


def create_device(path, ngpu=1, njob=1):
    shutil.rmtree(path, ignore_errors=True)
    os.mkdir(path)

    for i in range(ngpu):
        for j in range(njob):
            with open(path + '/%d_%d' % (i, j), 'a'):
                os.utime(path + '/%d_%d' % (i, j), None)


def get_device(path):
    device = os.listdir(path)[0]
    os.remove(path + '/' + device)
    return device


def return_device(path, device):
    with open(path + '/' + device, 'a'):
        os.utime(path + '/' + device, None)

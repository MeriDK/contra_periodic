import os
import sys
import joblib
import torch.multiprocessing as mp
from torch.multiprocessing import current_process
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data import MyDataset, preprocess, train_test_split
import numpy as np
import shutil
from train import train
from arg_parser import get_args
from utils import create_save_name, get_device, create_device, return_device
from model import get_network


# Global constants
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dint = torch.cuda.LongTensor
    map_loc = 'cuda:0'
else:
    dtype = torch.FloatTensor
    dint = torch.LongTensor
    map_loc = 'cpu'


def load_data(args):
    if 'asassn' in args.filename:
        args.max_sample = 20000

    if args.n_test == 1:
        lengths = [args.L]
    else:
        lengths = np.linspace(16, args.L * 2, args.n_test).astype(np.int)

        if args.L not in lengths:
            lengths = np.sort(np.append(lengths, args.L))

    data = joblib.load('data/{}'.format(args.filename))

    return data, lengths


def sanitize_data(data, args):
    # sanity check on dataset
    for lc in data:
        positive = lc.errors > 0
        positive *= lc.errors < 99
        lc.times = lc.times[positive]
        lc.measurements = lc.measurements[positive]
        lc.errors = lc.errors[positive]

    if 'macho' in args.filename:
        for lc in data:
            if 'LPV' in lc.label:
                lc.label = "LPV"

    # Generate a list all labels for train/test split
    unique_label, count = np.unique([lc.label for lc in data], return_counts=True)
    use_label = unique_label[count >= args.min_sample]

    n_classes = len(use_label)
    new_data = []
    for cls in use_label:
        class_data = [lc for lc in data if lc.label == cls]
        new_data.extend(class_data[:min(len(class_data), args.max_sample)])
    data = new_data

    all_label_string = [lc.label for lc in data]
    unique_label, count = np.unique(all_label_string, return_counts=True)
    print('------------before segmenting into L={}------------'.format(args.L))
    print(unique_label)
    print(count)
    convert_label = dict(zip(use_label, np.arange(len(use_label))))
    all_labels = np.array([convert_label[lc.label] for lc in data])

    n_inputs = 3 if args.use_error else 2

    return data, all_labels, n_classes, n_inputs, convert_label


def process_data(split, args, n_inputs, scales_all=None):
    X_list = [np.c_[chunk.times, chunk.measurements, chunk.errors] for chunk in split]
    periods = np.array([lc.p for lc in split])
    label = np.array([convert_label[chunk.label] for chunk in split])

    x, means, scales = preprocess(np.array(X_list), periods, use_error=args.use_error)

    print('shape of the dataset array:', x.shape)

    if scales_all is not None:
        mean_x = scales_all[0][:-1]
        std_x = scales_all[1][:-1]
    else:
        mean_x = x.reshape(-1, n_inputs).mean(axis=0)
        std_x = x.reshape(-1, n_inputs).std(axis=0)

    x -= mean_x
    x /= std_x
    if args.varlen_train:
        x = np.array(X_list)
    if args.two_phase:
        x = np.concatenate([x, x], axis=1)
    x = np.swapaxes(x, 2, 1)

    aux = np.c_[means, scales, np.log10(periods)]

    if args.use_meta and split[0].metadata is not None:
        metadata = np.array([lc.metadata for lc in split])  # Metadata must have same dimension!
        aux = np.c_[aux, metadata]                          # Concatenate metadata
        print('metadata will be used as auxiliary inputs.')

    if scales_all is not None:
        aux_mean = scales_all[2]
        aux_std = scales_all[3]
    else:
        aux_mean = aux.mean(axis=0)
        aux_std = aux.std(axis=0)

    aux -= aux_mean
    aux /= aux_std

    if scales_all is None:
        scales_all = np.array([np.append(mean_x, 0), np.append(std_x, 0), aux_mean, aux_std])

    return x, label, aux, scales_all


def create_dataloader(x, aux, label, batch_size, shuffle, drop_last, pin_memory=True):
    dset = MyDataset(x, aux, label)
    loader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, pin_memory=pin_memory)
    return loader


def train_model(mdl, args, name, train_loader, val_loader, test_loader, scales_all):
    if not args.no_log:
        import wandb
        wandb.init(project=args.project_name, config=args, name=name)
        wandb.watch(mdl)
    if not args.test:
        if args.retrain:
            mdl.load_state_dict(torch.load(name + '.pth', map_location=map_loc))
            args.lr *= 0.01
        optimizer = optim.Adam(mdl.parameters(), lr=args.lr)
        torch.manual_seed(args.seed)
        train(mdl, optimizer, train_loader, val_loader, test_loader, args.max_epoch,
              print_every=args.print_every, save=True, filename=name+args.note, patience=args.patience,
              early_stopping_limit=args.early_stopping, use_tqdm=True, scales_all=scales_all, clip=args.clip,
              retrain=args.retrain, decay_type=args.decay_type, monitor='accuracy', log=not args.no_log,
              perm=args.permute)


def train_helper(param):
    global map_loc
    train_index, test_index, name = param

    train_split = [chunk for i in train_index for chunk in data[i].split(args.L, args.L) if data[i].label is not None]
    test_split = [chunk for i in test_index for chunk in data[i].split(args.L, args.L) if data[i].label is not None]

    for lc in train_split:
        lc.period_fold()

    for lc in test_split:
        lc.period_fold()

    unique_label, count = np.unique([lc.label for lc in train_split], return_counts=True)
    print('------------after segmenting into L={}------------'.format(args.L))
    print(unique_label)
    print(count)

    x_train, label_train, aux_train, scales_all = process_data(train_split, args, n_inputs)
    x_test, label_test, aux_test, _ = process_data(test_split, args, n_inputs)

    if not args.varlen_train:
        scales_all = None
    else:
        np.save(name + '_scales.npy', scales_all)

    train_idx, val_idx = train_test_split(label_train, 1 - args.frac_valid, -1)

    if args.ngpu < 0:
        torch.cuda.set_device(int(-1*args.ngpu))
        map_loc = 'cuda:{}'.format(int(-1*args.ngpu))

    print('Using ', torch.cuda.current_device())
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    sys.stdout = sys.__stdout__

    train_loader = create_dataloader(x_train[train_idx], aux_train[train_idx], label_train[train_idx], args.train_batch,
                                     shuffle=True, drop_last=True)
    val_loader = create_dataloader(x_train[val_idx], aux_train[val_idx], label_train[val_idx], batch_size=128,
                                   shuffle=False, drop_last=False)
    test_loader = create_dataloader(x_test, aux_test, label_test, batch_size=128, shuffle=False, drop_last=False,
                                    pin_memory=True)

    mdl = get_network(n_inputs, n_classes, args, dtype)
    train_model(mdl, args, name, train_loader, val_loader, test_loader, scales_all)

    # load the model with the best validation accuracy for testing on the test set
    mdl.load_state_dict(torch.load(name + args.note + '.pth', map_location=map_loc))

    # Evaluate model on sequences of different length
    accuracy_length, accuracy_class_length = evaluate_model(mdl, args, lengths, test_loader, dtype)

    if args.ngpu > 1:
        return_device(path, device)
    return accuracy_length, accuracy_class_length


def evaluate_model(mdl, args, lengths, test_loader, dtype):
    accuracy_length = np.zeros(len(lengths))
    accuracy_class_length = np.zeros(len(lengths))
    mdl.eval()
    with torch.no_grad():
        for j, length in enumerate(lengths):

            softmax = torch.nn.Softmax(dim=1)
            predictions = []
            ground_truths = []
            for i, d in enumerate(test_loader):
                x, aux_, y = d
                logprob = mdl(x.type(dtype), aux_.type(dtype))
                predictions.extend(list(np.argmax(softmax(logprob).detach().cpu(), axis=1)))
                ground_truths.extend(list(y.numpy()))

            predictions = np.array(predictions)
            ground_truths = np.array(ground_truths)

            accuracy_length[j] = (predictions == ground_truths).mean()
            accuracy_class_length[j] = np.array(
                [(predictions[ground_truths == l] == ground_truths[ground_truths == l]).mean()
                 for l in np.unique(ground_truths)]).mean()

    return accuracy_length, accuracy_class_length


if __name__ == '__main__':

    args = get_args()
    save_name = create_save_name(args)

    if current_process().name != 'MainProcess':
        if args.njob > 1 or args.ngpu > 1:
            path = 'device' + save_name + args.note
            device = get_device(path)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(device[0])
    else:
        print('save filename:')
        print(save_name)

    data, lengths = load_data(args)
    data, all_labels, n_classes, n_inputs, convert_label = sanitize_data(data, args)

    jobs = []
    np.random.seed(args.seed)
    for i in range(args.K):
        if args.K == 1:
            i = args.pseed
        trains, tests = train_test_split(all_labels, train_size=args.frac_train, random_state=i)
        jobs.append((trains, tests, '{}/{}-{}'.format(args.path, save_name, i)))
    try:
        os.mkdir(args.path)
    except:
        pass
    if args.ngpu <= 1 and args.njob == 1:
        results = []
        for j in jobs:
            results.append(train_helper(j))
    else:
        create_device('device'+save_name+args.note, args.ngpu, args.njob)
        ctx = mp.get_context('spawn')
        with ctx.Pool(args.ngpu * args.njob) as p:
            results = p.map(train_helper, jobs)
        shutil.rmtree('device' + save_name+args.note)
    results = np.array(results)
    results_all = np.c_[lengths, results[:, 0, :].T]
    results_class = np.c_[lengths, results[:, 1, :].T]
    np.save('{}/{}{}-results.npy'.format(args.path, save_name, args.note), results_all)
    np.save('{}/{}{}-results-class.npy'.format(args.path, save_name, args.note), results_class)

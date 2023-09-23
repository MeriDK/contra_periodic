import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data import load_data
import numpy as np
from train import train, evaluate
from arg_parser import get_args
from utils import create_save_name, get_model
import wandb


def main():
    args = get_args()
    save_name = create_save_name(args)
    os.makedirs(args.path, exist_ok=True)
    wandb.init(project=args.project_name, config=args, name=save_name)

    print('Using ', torch.cuda.current_device())
    dtype, dint, map_loc = torch.cuda.FloatTensor, torch.cuda.LongTensor, 'cuda:0'

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_dataset, val_dataset, test_dataset, scales_all, info = load_data(args)
    n_inputs, n_classes = info['n_inputs'], info['n_classes']

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False, pin_memory=True)

    model = get_model(n_inputs, n_classes, args, dtype)
    wandb.watch(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train(model, optimizer, train_loader, val_loader, n_epoch=args.max_epoch, print_every=args.print_every,
          save=True, filename=save_name+args.note, early_stopping_limit=args.early_stopping, use_tqdm=True,
          clip=args.clip, retrain=args.retrain, decay_type=args.decay_type, monitor='accuracy', perm=args.permute)

    # load the model with the best validation accuracy for testing on the test set
    model.load_state_dict(torch.load(save_name + args.note + '.pth', map_location=map_loc))

    # TODO fix length
    # Evaluate model on sequences of different length
    accuracy_length, accuracy_class_length = evaluate(model, [200], test_loader, dtype)


if __name__ == '__main__':
    main()

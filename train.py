import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from tqdm import tqdm_notebook
from tqdm import tqdm as tqdm_
import wandb
from data import permute
from torch import from_numpy

# Define types depending on whether CUDA is available
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dint = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


def train_epoch(model, train_loader, optimizer, loss_fn, clip, scales_all, use_tqdm, perm, tqdm):
    model.train()
    train_loss = []
    predictions = []
    ground_truths = []

    for x, aux, y in tqdm(train_loader, disable=not use_tqdm):
        if scales_all is not None:
            x, aux = preprocess_data(x, aux, scales_all)
        if perm:
            x = permute(x)

        loss, prediction = train_step(model, x, aux, y, optimizer, loss_fn, clip)
        train_loss.append(loss.detach().cpu())
        predictions.extend(prediction)
        ground_truths.extend(list(y.numpy()))

    return np.array(train_loss).mean(), np.array(predictions), np.array(ground_truths)


def train_step(model, x, aux, y, optimizer, loss_fn, clip):
    logprob = model(x.type(dtype), aux.type(dtype))
    loss = loss_fn(logprob, y.type(dint))
    optimizer.zero_grad()
    loss.backward()

    if clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    optimizer.step()

    softmax = torch.nn.Softmax(dim=1)
    prediction = list(np.argmax(softmax(logprob).detach().cpu(), axis=1))

    return loss, prediction


def validate_epoch(model, val_loader, loss_fn, scales_all, use_tqdm, perm, tqdm):
    model.eval()
    val_loss = []
    predictions = []
    ground_truths = []

    for x, aux, y in tqdm(val_loader, disable=not use_tqdm):
        if scales_all is not None:
            x, aux = preprocess_data(x, aux, scales_all)
        if perm:
            x = permute(x)

        loss, prediction = validate_step(model, x, aux, y, loss_fn)
        val_loss.extend([loss.detach().cpu()] * x.shape[0])
        predictions.extend(prediction)
        ground_truths.extend(list(y.numpy()))

    return np.array(val_loss).mean(), np.array(predictions), np.array(ground_truths)


def validate_step(model, x, aux, y, loss_fn):
    logprob = model(x.type(dtype), aux.type(dtype))
    loss = loss_fn(logprob, y.type(dint))
    softmax = torch.nn.Softmax(dim=1)
    prediction = list(np.argmax(softmax(logprob).detach().cpu(), axis=1))

    return loss, prediction


def times_to_lags(x):
    lags = x[:, 1:] - x[:, :-1]
    return lags


# preprocess into (dt, f) representation.
# for variable length training only
def preprocess(X_raw):
    N, F, L = X_raw.shape
    X = torch.zeros((N, 2, L-1))
    X[:, 0, :] = times_to_lags(X_raw[:, 0, :]) / torch.max(X_raw[:, 0, :], dim=1)[0][:,None]
    X[:, 1, :] = X_raw[:, 1, :][:,:-1]
    means = X_raw[:, 1, :].mean(dim=1).float()
    scales = X_raw[:, 1, :].std(dim=1).float()
    X[:, 1, :] -= means[:, None]
    X[:, 1, :] /= scales[:, None]
    return X, means, scales


def preprocess_data(x, aux, scales_all):
    mean_x, std_x, aux_mean, aux_std = scales_all
    mean_x = mean_x[:-1]
    std_x = std_x[:-1]

    L = np.random.randint(16, x.shape[2])
    indexes = np.sort(np.random.choice(range(x.shape[2]), L, replace=False))
    x = x[:, :, indexes]
    x, means, scales = preprocess(x)
    x -= mean_x[None, :, None]
    x /= std_x[None, :, None]
    aux[:, 0] = (means - aux_mean[0]) / aux_std[0]
    aux[:, 1] = (scales - aux_mean[1]) / aux_std[1]

    return x, aux


def load_previous_metrics(filename, retrain):
    if retrain:
        val_accuracy = list(np.load(filename + '-CONVERGENCE-val.npy'))
        train_accuracy = list(np.load(filename + '-CONVERGENCE-train.npy'))
        val_losses = list(np.load(filename + '-CONVERGENCE-val-loss.npy'))
        train_losses = list(np.load(filename + '-CONVERGENCE-train-loss.npy'))
    else:
        train_accuracy = []
        val_accuracy = []
        train_losses = []
        val_losses = []

    return train_accuracy, val_accuracy, train_losses, val_losses


def get_lr_scheduler(optimizer, decay_type, patience, threshold):
    if decay_type == 'plateau':
        return ReduceLROnPlateau(optimizer, factor=0.1, patience=patience,
                                 cooldown=0, verbose=True, threshold=threshold)
    elif decay_type == 'exp':
        return ExponentialLR(optimizer, 0.85)


def log_metrics(log, train_loss, val_loss, train_accuracy, accuracy):
    if log:
        wandb.log({
            "Train Loss": train_loss,
            "Val Loss": val_loss,
            "Train Acc": train_accuracy[-1] * 100,
            "Val Acc": accuracy * 100
        })


def save_model(model, save, filename, epoch, val_loss=None, accuracy=None):
    if save:
        torch.save(model.state_dict(), filename+'.pth')
        if val_loss is not None:
            print(f'Saved: epoch:{epoch}: val_loss = {val_loss:.4f}')
        elif accuracy is not None:
            print(f'Saved: epoch:{epoch}: accuracy = {accuracy*100:.2f}')


def train(model, optimizer, train_loader, val_loader, test_loader, n_epoch, eval_after=1e5, patience=10, min_lr=0.00001,
          filename='model', save=False, monitor='accuracy', print_every=-1, early_stopping_limit=1e5,
          threshold=0.1, use_tqdm=False, jupyter=False, scales_all=None, clip=-1, retrain=False, decay_type='plateau',
          log=False, perm=True):

    tqdm = tqdm_notebook if jupyter else tqdm_
    # TODO create additional vars mean, std, etc instead of scales_all
    # scales_all = from_numpy(scales_all).float() if scales_all is not None else [1, 1, 1, 1]
    scales_all = None
    loss_fn = nn.CrossEntropyLoss()
    lr_scheduler = get_lr_scheduler(optimizer, decay_type, patience, threshold)
    train_accuracy, val_accuracy, train_losses, val_losses = load_previous_metrics(filename, retrain)

    print('------------begin training---------------')

    for epoch in tqdm(range(n_epoch), disable=not use_tqdm):
        train_loss, train_predictions, train_ground_truths = train_epoch(model, train_loader, optimizer, loss_fn, clip,
                                                                         scales_all, use_tqdm, perm, tqdm)
        val_loss, val_predictions, val_ground_truths = validate_epoch(model, val_loader, loss_fn, scales_all, use_tqdm,
                                                                      perm, tqdm)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accuracy.append((train_predictions == train_ground_truths).mean())
        val_accuracy.append((val_predictions == val_ground_truths).mean())

        lr_scheduler.step(train_loss if decay_type == 'plateau' else None)
        log_metrics(log, train_loss, val_loss, train_accuracy, val_accuracy[-1])

        if print_every != -1 and epoch % print_every == 0:
            print(f'epoch:{epoch}: train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f},'
                  f'accuracy = {val_accuracy[-1] * 100:.2f}')

        if monitor == 'val_loss' and val_loss <= min(train_losses):
            save_model(model, save, filename, epoch, val_loss=val_loss)
        elif monitor == 'accuracy' and val_accuracy[-1] >= max(val_accuracy):
            save_model(model, save, filename, epoch, accuracy=val_accuracy[-1])

        if 0 < early_stopping_limit < epoch - np.argmax(val_accuracy):
            print(f'Metric did not improve for {early_stopping_limit} rounds')
            print(f'Early stopping at epoch {epoch}')
            break

    return train_losses, val_losses, max(train_accuracy), max(val_accuracy)

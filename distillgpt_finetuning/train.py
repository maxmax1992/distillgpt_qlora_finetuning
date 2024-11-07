import time
import os
import numpy as np
import pandas as pd
from typing import List, Dict
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from distillgpt_finetuning.utils.data_utils import get_train_data
from distillgpt_finetuning.utils.dataloading import get_train_and_val_loaders
from distillgpt_finetuning.utils.model import get_model
import argparse
import torch


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        t1 = time.time()
        data = [x[0] for x in batch]
        target = torch.tensor([x[1] for x in batch])
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        pred = output.detach().argmax(dim=1, keepdims=True)
        accuracy = pred.eq(target.view_as(pred)).sum().item()/len(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        elapsed = time.time() - t1
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSample accuracy: {:.3f}\tTime-per-batch: {:.2f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), accuracy, elapsed))
            if args.dry_run:
                break


def test(args, model, device, test_loader) -> float:
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for _, batch in enumerate(test_loader):

            data = [x[0] for x in batch]
            target = torch.tensor([x[1] for x in batch])
            target = target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            if args.dry_run:
                break

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss


def get_id_to_probs_map(args, loader, model) -> List[np.array]:
    numpy_array_list = []
    for _, batch in tqdm(enumerate(loader)):
        data = [x[0] for x in batch]
        output = model(data)
        pred = output.detach()
        softmax_preds = F.softmax(pred, dim=1).cpu().numpy()
        ids = [datapoint['id'] for datapoint in data]
        ids_col = np.array(ids).reshape((-1, 1))
        np_chunk = np.column_stack((ids_col, softmax_preds))
        numpy_array_list.append(np_chunk)
        
        if args.dry_run:
            return numpy_array_list
    return numpy_array_list


def make_sample_predictions_to_csv(args, model, train_loader, val_loader):
    """
        Create sample predictions over all data and save to ./output_csv with each class probabilities
    """
    np_array_final = None
    with torch.no_grad():
        numpy_arrays = get_id_to_probs_map(args, train_loader, model)
        numpy_arrays += get_id_to_probs_map(args, val_loader, model)
        for numpy_arr in numpy_arrays:
            if np_array_final is None:
                np_array_final = numpy_arr
            else:
                np_array_final = np.vstack((np_array_final, numpy_arr))
        pd.DataFrame(data=np_array_final, columns=['id', 'c0', 'c1', 'c2', 'c3']).to_csv('./part_2_predictions.csv', index=False)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training configuration arguments")

    # Adding arguments
    parser.add_argument('--train_frac', type=float, default=0.8, help='Fraction of data to use for training')
    parser.add_argument('--log_interval', type=int, default=10, help='Interval for logging progress')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loading')
    parser.add_argument('--dry_run', action='store_true', help='Run without saving changes')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate for training')
    parser.add_argument('--gamma', type=float, default=0.7, help='Learning rate decay factor')
    parser.add_argument('--n_epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=4, help='Random seed for reproducibility')
    parser.add_argument('--n_classes', type=int, default=4, help='Number of classes to use')
    parser.add_argument('--model_save_dir', type=str, default='./models', help='Directory to save trained models')

    return parser.parse_args()

def main(args):
    inputs_path ='./sample_data/sample_data.json'
    targets_path = "./sample_data/sample_targets.csv"
    df_preprocessed, targets = get_train_data(inputs_path, targets_path)
    torch.manual_seed(args.seed)
    model = get_model(args.n_classes)

    train_loader, val_loader = get_train_and_val_loaders(df_preprocessed, targets, train_frac=args.train_frac,
                                                         batch_size=args.batch_size)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    device = model.device
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    if not os.path.exists(args.model_save_dir):
        os.mkdir(args.model_save_dir)

    for epoch in range(1, args.n_epochs+1):
        train(args, model, device, train_loader, optimizer, epoch)
        avg_val_loss = test(args, model, device, val_loader)
        save_model_name = f"e_{epoch}_{avg_val_loss:.4f}_model.pth"
        model_save_path = os.path.join(args.model_save_dir, save_model_name)
        torch.save(model, model_save_path)
        if args.dry_run:
            break
        scheduler.step()
    make_sample_predictions_to_csv(args, model, train_loader, val_loader)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)

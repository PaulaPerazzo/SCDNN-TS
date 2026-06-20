# ==============================================================================
# Baseline 1D CNN Training Script for ECG classification (PTB-XL)
# ==============================================================================

import argparse
import os
import pickle
import random
import time
import copy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from tqdm import tqdm
import wandb

# --- SUAS IMPORTAÇÕES CUSTOMIZADAS ---
from Adam import Adam as newAdam
from dataset import data_prep
from loss_library import FocalLoss
from notification import notificar_ntfy

# --- IMPORTANDO O MODELO DO SEU ARQUIVO ---
from CNN import BaselineCNN1D


# ==========================================
# 1. CONFIGURAÇÕES INICIAIS E DISPOSITIVO
# ==========================================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU ({torch.cuda.get_device_name(device)})")
else:
    device = torch.device("cpu")

torch.cuda.empty_cache()

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==========================================
# 2. LOOP DE TREINAMENTO E VALIDAÇÃO
# ==========================================
def start_train(args, model, train_loader, test_loader, device):
    optimizer = newAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 700], gamma=0.1)

    if args.criterion == 'focalloss':
        criterion = FocalLoss(gamma=2.0)
        print("Using Focal Loss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using Cross Entropy Loss")

    print("Start training baseline model...")
    print("---------------------------")
    time_start = time.time()

    def train(model, criterion, optimizer, train_loader, device):
        epoch_loss = 0.0
        model.train()
        for data in train_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.reshape(-1).to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        return model, optimizer, epoch_loss / len(train_loader)

    @torch.no_grad()
    def test(model, criterion, test_loader, device):
        model.eval()
        test_loss = 0
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.reshape(-1).to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
        return model, test_loss / len(test_loader)

    @torch.no_grad()
    def get_acc(model, test_loader):
        model.eval()
        total_acc = 0
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.reshape(-1).to(device)
            
            outputs = model(inputs)
            final_outputs = torch.argmax(outputs, dim=1)
            total_acc += torch.sum(final_outputs == labels).item()
        return total_acc / len(test_loader.dataset)

    train_loss_list, test_loss_list, test_acc_list = [], [], []
    best_loss = float('inf')
    best_acc = 0
    best_model = None
    best_epoch = 0
    patience_counter = 0

    for epoch in tqdm(range(args.epochs)):
        model, optimizer, epoch_loss = train(model, criterion, optimizer, train_loader, device)
        train_loss_list.append(epoch_loss)
        scheduler.step()

        with torch.no_grad():
            model, test_loss = test(model, criterion, test_loader, device)
            test_loss_list.append(test_loss)

        test_acc = get_acc(model, test_loader)
        test_acc_list.append(test_acc)

        if test_loss < best_loss:
            best_loss = test_loss
            best_acc = test_acc
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        if epoch % args.print_step == 0:
            print(f"Epoch:{epoch}\tTrain Loss:{epoch_loss:.4f}\tVal Loss:{test_loss:.4f}\tEpoch Acc:{test_acc:.4f}")
        
        wandb.log({"acc": test_acc, "train_loss": epoch_loss, "val_loss": test_loss})

    print(f"Finished Training in {args.epochs} epochs. Best Accuracy: {best_acc:.4f} at epoch {best_epoch}")
    print("Total training time: ", time.time() - time_start)
    
    try:
        notificar_ntfy("Sua CNN Base acabou de treinar!")
    except:
        pass

    if args.save_model:
        model_path = os.path.join(args.model_dir, f"BaselineCNN_LR={args.lr}_Epochs={args.epochs}.pth")
        torch.save(best_model.state_dict(), model_path)

    return best_model, train_loss_list, test_loss_list, test_acc_list

# ==========================================
# 3. AVALIAÇÃO E GRÁFICOS
# ==========================================
@torch.no_grad()
def start_test(model, dataloader, args):
    print("Start testing model...")
    print("---------------------------")
    model.eval()
    label = []
    test_label = []

    for data in dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs_label = torch.argmax(outputs, dim=1)

        label.append(outputs_label)
        test_label.append(labels)

    label = torch.cat(label).cpu().detach().numpy()
    test_label = torch.cat(test_label).cpu().detach().numpy()

    report = classification_report(test_label, label, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(os.path.join(args.model_dir, "report_baseline.csv"))

    return label

def plot_result(train_loss, test_loss, test_acc, args):
    path = os.path.join(args.model_dir, "train_log_baseline.pkl")
    with open(path, "wb") as f:
        pickle.dump([train_loss, test_loss, test_acc], f)

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_loss)
    plt.yscale("log")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("Train Loss")

    plt.subplot(1, 3, 2)
    plt.plot(test_loss)
    plt.yscale("log")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("Val Loss")

    plt.subplot(1, 3, 3)
    plt.plot(test_acc)
    plt.xlabel("epochs")
    plt.ylabel("ACC")
    plt.title("Val Accuracy")

    figpath = os.path.join(args.model_dir, "baseline_result_curve.png")
    plt.savefig(figpath, dpi=300)
    print("Curvas da Baseline salvas.")

# ==========================================
# 4. EXECUÇÃO PRINCIPAL
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/superclass/shuffle-data")
    parser.add_argument("--task", type=str, default="superclass")
    parser.add_argument("--model_dir", type=str, default="superclass/")
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--save_model", action="store_true", default=True)
    parser.add_argument("--criterion", default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--print_step", type=int, default=10)
    parser.add_argument("--task_num", type=int, default=1)
    parser.add_argument("--patience", type=int, default=15, help="Patience for Early Stopping")
    args = parser.parse_args()

    if args.seed:
        setup_seed(args.seed)

    path = "./data_baseline/" + args.task + "/" + str(args.task_num)
    if not os.path.exists(path):
        os.makedirs(path)

    args.model_dir = path + "/"
    
    train_loader, test_loader, val_loader = data_prep(args.task, args.batch_size)
    print("Training set Prepared!")
    print("---------------------------")

    task_classes = {
        "superclass": 5,
        "subclass": 23,
        "rthym": 12,
        "form": 19
    }
    
    num_classes = task_classes.get(args.task, 5)
    if args.task not in task_classes:
        print("Warning: Incorrect task name, defaulting to 5 classes.")

    # Instanciando o modelo importado do CNN.py
    model = BaselineCNN1D(in_channels=12, num_classes=num_classes).to(device)

    wandb_name = args.task + "-baseline-cnn"
    wandb.init(project="tnnls",entity='mpperazzosampaio-cin-ufpe', name=wandb_name)
    wandb.config = {
        "task name": args.task,
        "task number": args.task_num,
        "model_type": "Baseline_1D_CNN"
    }

    # Treinamento
    best_model, train_loss, test_loss, test_acc = start_train(args, model, train_loader, test_loader, device)

    # Gráficos
    plot_result(train_loss, test_loss, test_acc, args)

    # Teste final
    y_pred = start_test(best_model, test_loader, args)
    np.save(os.path.join(path, "y_pred_baseline.npy"), y_pred)
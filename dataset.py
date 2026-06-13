import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import distributed as dist

import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

import random
import numpy as np
import torch

def set_all_seeds(seed=42):
    """
    Fixa todas as seeds para garantir a reprodutibilidade dos experimentos.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_all_seeds(42)

def get_balanced_subset(X, y, samples_per_class):
    """
    Filtra X e y para retornar um subconjunto balanceado.
    (Esta função será ignorada na execução completa)
    """
    unique_classes = np.unique(y)
    balanced_indices = []

    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        n_samples = min(samples_per_class, len(cls_indices))
        selected_indices = np.random.choice(cls_indices, n_samples, replace=False)
        balanced_indices.extend(selected_indices)

    balanced_indices = np.array(balanced_indices)
    np.random.shuffle(balanced_indices)

    return X[balanced_indices], y[balanced_indices]


# ALTERAÇÃO AQUI: Parâmetros alterados para None por padrão
def data_prep(task, batch_size, train_samples=None, val_samples=None, test_samples=None):
    print("Data Loading start!")
    print("---------------------------")

    path = "./data/" + task + "/benchmarkdata/"

    P_X_train, P_X_test, P_X_val = (
        path + "x_train.npy",
        path + "x_test.npy",
        path + "x_val.npy",
    )
    P_y_train, P_y_test, P_y_val = (
        path + "y_train.npy",
        path + "y_test.npy",
        path + "y_val.npy",
    )

    X_train = np.load(P_X_train, allow_pickle=True)
    X_test = np.load(P_X_test, allow_pickle=True)
    X_val = np.load(P_X_val, allow_pickle=True)

    X_train = np.swapaxes(X_train, 1, 2)
    X_test = np.swapaxes(X_test, 1, 2)
    X_val = np.swapaxes(X_val, 1, 2)

    y_train = np.load(P_y_train, allow_pickle=True)
    y_test = np.load(P_y_test, allow_pickle=True)
    y_val = np.load(P_y_val, allow_pickle=True)

    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)
    y_val = np.argmax(y_val, axis=1)

    if train_samples is not None:
        print(f"Balanceando Treino para {train_samples} amostras/classe...")
        X_train, y_train = get_balanced_subset(X_train, y_train, train_samples)
        
    if test_samples is not None:
        print(f"Balanceando Teste para {test_samples} amostras/classe...")
        X_test, y_test = get_balanced_subset(X_test, y_test, test_samples)
        
    if val_samples is not None:
        print(f"Balanceando Validação para {val_samples} amostras/classe...")
        X_val, y_val = get_balanced_subset(X_val, y_val, val_samples)
    
    print("X_train shape after loading:", X_train.shape)
    print("X_test shape after loading:", X_test.shape)
    print("X_val shape after loading:", X_val.shape)

    print("total data: ", len(X_train) + len(X_val) + len(X_test))

    print("Data Loading completed!")
    print("---------------------------")

    def convert(label):
        converted_label = label.astype(np.float32).reshape(-1, 1)
        return converted_label

    class dataset(Dataset):
        def __init__(self, signal, label):
            self.data = signal
            self.label = label

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            ecg = torch.from_numpy(self.data[idx].astype(np.float32))
            target = torch.from_numpy(self.label[idx]).type(torch.long)
            return (ecg, target)

    converted_y_train = convert(y_train)
    converted_y_test = convert(y_test)
    converted_y_val = convert(y_val)

    trainset = dataset(signal=X_train, label=converted_y_train)
    testset = dataset(signal=X_test, label=converted_y_test)
    valset = dataset(signal=X_val, label=converted_y_val)

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, pin_memory=False, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, pin_memory=False, shuffle=False, num_workers=4)
    val_loader = DataLoader(dataset=valset, batch_size=batch_size, pin_memory=False, shuffle=False, num_workers=4)

    return train_loader, test_loader, val_loader

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
    # Fixa a seed do random nativo do Python
    random.seed(seed)
    
    # Fixa a seed do Numpy (afeta o nosso get_balanced_subset)
    np.random.seed(seed)
    
    # Fixa as seeds do PyTorch (afeta a inicialização de pesos da sua CNN e os DataLoaders)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Se estiver usando múltiplas GPUs
    
    # Configurações adicionais para forçar o determinismo nas operações convolucionais da GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Chame a função antes de carregar os dados ou instanciar o modelo
set_all_seeds(42)

def get_balanced_subset(X, y, samples_per_class):
    """
    Filtra X e y para retornar um subconjunto balanceado com 'samples_per_class' 
    amostras por classe.
    """
    unique_classes = np.unique(y)
    balanced_indices = []

    for cls in unique_classes:
        # Encontra todos os índices pertencentes à classe atual
        cls_indices = np.where(y == cls)[0]
        
        # Garante que não vamos pedir mais amostras do que a classe possui
        n_samples = min(samples_per_class, len(cls_indices))
        
        # Seleciona aleatoriamente 'n_samples' índices sem reposição
        selected_indices = np.random.choice(cls_indices, n_samples, replace=False)
        balanced_indices.extend(selected_indices)

    # Converte para array numpy
    balanced_indices = np.array(balanced_indices)
    
    # Embaralha os índices para que os dados não fiquem agrupados e ordenados por classe
    np.random.shuffle(balanced_indices)

    return X[balanced_indices], y[balanced_indices]

# Adicionado o parâmetro 'samples_per_class'
# Mantenha a função get_balanced_subset e set_all_seeds que fizemos anteriormente

def data_prep(task, batch_size, train_samples=1000, val_samples=200, test_samples=200):
    ### -------------------------------
    # load data and target
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
    ###

    X_train = np.load(P_X_train, allow_pickle=True)
    X_test = np.load(P_X_test, allow_pickle=True)
    X_val = np.load(P_X_val, allow_pickle=True)

    X_train = np.swapaxes(X_train, 1, 2)
    X_test = np.swapaxes(X_test, 1, 2)
    X_val = np.swapaxes(X_val, 1, 2)
    ###

    y_train = np.load(P_y_train, allow_pickle=True)
    y_test = np.load(P_y_test, allow_pickle=True)
    y_val = np.load(P_y_val, allow_pickle=True)

    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)
    y_val = np.argmax(y_val, axis=1)
    ###

    # ---------------------------------------------------------
    # APLICAÇÃO DO BALANCEAMENTO COM TAMANHOS DISTINTOS
    # ---------------------------------------------------------
    if train_samples is not None:
        print(f"Balanceando Treino para {train_samples} amostras/classe...")
        X_train, y_train = get_balanced_subset(X_train, y_train, train_samples)
        
    if test_samples is not None:
        print(f"Balanceando Teste para {test_samples} amostras/classe...")
        X_test, y_test = get_balanced_subset(X_test, y_test, test_samples)
        
    if val_samples is not None:
        print(f"Balanceando Validação para {val_samples} amostras/classe...")
        X_val, y_val = get_balanced_subset(X_val, y_val, val_samples)
    
    print("X_train shape after balancing:", X_train.shape)
    print("Data Loading completed!")
    print("---------------------------")

    ### -------------------------------
    # Resto da sua função de conversão, Dataset e DataLoaders permanece igual...
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

    # Lembre-se de adequar o batch_size das validações se o dataset for menor que ele
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, pin_memory=True, shuffle=True)
    test_loader = DataLoader(dataset=testset, batch_size=min(2048, len(testset)), pin_memory=True, shuffle=False)
    val_loader = DataLoader(dataset=valset, batch_size=min(1024, len(valset)), pin_memory=True, shuffle=False)

    return train_loader, test_loader, val_loader

### --------------------------------
## version for distributed training

# trainset = dataset(signal = X_train, label = converted_y_train, local_rank=local_rank)
# testset = dataset(signal = X_test, label = converted_y_test, local_rank=local_rank)
# valset = dataset(signal = X_val, label = converted_y_val, local_rank=local_rank)

# testset_org = dataset(signal = X_test_org, label = converted_y_test_org, local_rank=local_rank)

# # define sampler for each GPU -- for shuffle on each epoch
# sampler = torch.utils.data.distributed.DistributedSampler(trainset)

# if parallel_type is 'None' or 'Dataparallel':
#     # for data parallel and non-parallel

#     train_loader = DataLoader(dataset=trainset, batch_size=batch_size, pin_memory = True, shuffle=True)
#     test_loader = DataLoader(dataset=testset, batch_size=batch_size, pin_memory = True, shuffle=False)
#     val_loader = DataLoader(dataset=valset, batch_size=1024, pin_memory = True, shuffle=False)

#     test_loader_org = DataLoader(dataset=testset_org, batch_size=1024, pin_memory = True, shuffle=False)
# else:
#     # for model distributed training

#     sampler = torch.utils.data.distributed.DistributedSampler(trainset)

#     train_loader = DataLoader(trainset,
#                       batch_size=batch_size,
#                       shuffle=False,
#                       pin_memory=True,
#                       drop_last=True,
#                       sampler=sampler)
#     test_loader = DataLoader(dataset=testset, batch_size=batch_size, pin_memory = True, shuffle=False)

# return train_loader, test_loader, label_encoder, y_test, sampler

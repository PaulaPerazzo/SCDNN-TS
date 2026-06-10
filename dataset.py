import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import wfdb
import ast

# Função auxiliar para carregar os sinais usando a biblioteca WFDB
def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    
    # Extrai apenas os sinais (ignorando os metadados do rdsamp por agora)
    data = np.array([signal for signal, meta in data])
    return data

def data_prep( batch_size, start_date=None, end_date=None, sampling_rate=100):
    print('Iniciando o carregamento dos dados brutos...')
    print('---------------------------')
    dataset_path = 'data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
    # 1. Carregar os metadados
    df = pd.read_csv(dataset_path + 'ptbxl_database.csv', index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x)) # Converte string de dict para dict real

    # 2. Filtrar por data (Opcional)
    if start_date and end_date:
        df['record_date'] = pd.to_datetime(df['record_date'])
        mask = (df['record_date'] >= start_date) & (df['record_date'] <= end_date)
        df = df.loc[mask]
        print(f"Dados filtrados entre {start_date} e {end_date}. Amostras restantes: {len(df)}")

    # 3. Fazer o Split Padrão do PTB-XL (Folds de 1 a 8 = Treino, 9 = Validação, 10 = Teste)
    # Isso garante que você está usando o mesmo benchmark oficial
    train_df = df[df.strat_fold <= 8]
    val_df = df[df.strat_fold == 9]
    test_df = df[df.strat_fold == 10]



    # 5. CRIAR LABELS REAIS PARA A TASK 'SUPERCLASS'
    print("Processando diagnósticos do PTB-XL...")
    
    # Carrega o dicionário oficial de diagnósticos do PTB-XL
    # (Certifique-se de que o arquivo scp_statements.csv está na pasta dataset_path)
    agg_df = pd.read_csv(dataset_path + 'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1] # Filtra apenas os statements que são diagnósticos
    
    # Função para extrair a superclasse a partir dos códigos SCP do paciente
    def get_superclass(y_dic):
        for key in y_dic.keys():
            if key in agg_df.index:
                # Retorna a primeira superclasse válida que encontrar
                return agg_df.loc[key].diagnostic_class
        return None # Retorna None se não for um diagnóstico principal

    # Aplica a extração para todos os pacientes
    df['superclass'] = df.scp_codes.apply(get_superclass)
    
    # Mapeia as 5 classes oficiais para números inteiros (0 a 4)
    # NORM (Normal), MI (Infarto), STTC (Alt. ST/T), CD (Distúrbio de Condução), HYP (Hipertrofia)
    class_map = {'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4}
    df['label'] = df['superclass'].map(class_map)
    
    # Remove os pacientes que não possuem nenhuma dessas 5 superclasses
    tamanho_antes = len(df)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    print(f"Pacientes com diagnósticos válidos: {len(df)} de {tamanho_antes}")

    # --- AGORA REFAZEMOS O SPLIT COM OS DADOS LIMPOS ---
    train_df = df[df.strat_fold <= 8]
    val_df = df[df.strat_fold == 9]
    test_df = df[df.strat_fold == 10]

    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values

        # 4. Carregar os Sinais (Isso pode demorar alguns minutos dependendo do tamanho do filtro)
    print("Carregando sinais de Treino...")
    X_train = load_raw_data(train_df, sampling_rate, dataset_path)
    print("Carregando sinais de Validação...")
    X_val = load_raw_data(val_df, sampling_rate, dataset_path)
    print("Carregando sinais de Teste...")
    X_test = load_raw_data(test_df, sampling_rate, dataset_path)

    # Swapaxes para adequar ao formato do PyTorch (Batch, Channels, Length)
    X_train = np.swapaxes(X_train, 1, 2)
    X_test = np.swapaxes(X_test, 1, 2)
    X_val = np.swapaxes(X_val, 1, 2)

    print('Carregamento concluído!')
    print('---------------------------')

    # -------------------------------
    # Conversor e Definição do Dataset (Mantidos do seu código original)
    def convert(label):
        converted_label = label.astype(np.float32).reshape(-1,1)
        return converted_label

    class ECGDataset(Dataset):
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

    # --------------------------------
    # converter target para tensor e criar Datasets
    trainset = ECGDataset(signal=X_train, label=convert(y_train))
    valset = ECGDataset(signal=X_val, label=convert(y_val))
    testset = ECGDataset(signal=X_test, label=convert(y_test))

    # --------------------------------
    # criar DataLoaders
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, pin_memory=True, shuffle=True)
    val_loader = DataLoader(dataset=valset, batch_size=1024, pin_memory=True, shuffle=False)
    test_loader = DataLoader(dataset=testset, batch_size=2048, pin_memory=True, shuffle=False)

    return train_loader, test_loader, val_loader

# --- COMO USAR O NOVO CÓDIGO ---
# caminho_do_dataset = '/caminho/para/seu/ptbxl/' (Onde estão os arquivos CSV e as pastas records100)
# train_loader, test_loader, val_loader = data_prep_from_raw(caminho_do_dataset, batch_size=64, start_date='1990-01-01', end_date='1995-12-31')
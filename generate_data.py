import os
import ast
import wfdb
import numpy as np
import pandas as pd

# =====================================================
# CONFIGURAÇÃO
# =====================================================

PTBXL_PATH = (
    "./data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
)
OUTPUT_DIR = "./data/superclass/benchmarkdata"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# CARREGAR ECGs
# =====================================================


def load_raw_data(df, sampling_rate, path):

    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]

    data = np.array([signal for signal, meta in data])

    return data


print("Loading metadata...")

Y = pd.read_csv(os.path.join(PTBXL_PATH, "ptbxl_database.csv"), index_col="ecg_id")

Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)

print("Loading ECG signals...")

X = load_raw_data(Y, 100, PTBXL_PATH)

print("Original X shape:", X.shape)
print("Original Y size :", len(Y))

# =====================================================
# MAPEAMENTO SCP -> SUPERCLASS
# =====================================================

agg_df = pd.read_csv(os.path.join(PTBXL_PATH, "scp_statements.csv"), index_col=0)

agg_df = agg_df[agg_df.diagnostic == 1]


def aggregate_diagnostic(y_dic):

    tmp = []

    for key in y_dic.keys():

        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)

    return list(set(tmp))


Y["diagnostic_superclass"] = Y.scp_codes.apply(aggregate_diagnostic)

# =====================================================
# FILTRAR APENAS ECGs COM UMA CLASSE
# =====================================================

single_mask = (Y.diagnostic_superclass.apply(len) == 1).values

X = X[single_mask]
Y = Y[single_mask].copy()

print("After single-label filter:")
print("X:", X.shape)
print("Y:", len(Y))

# =====================================================
# CLASSES UTILIZADAS PELO MODELO
# =====================================================

classes = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP",
]

class_to_idx = {c: i for i, c in enumerate(classes)}

Y["label"] = Y.diagnostic_superclass.apply(lambda x: x[0])

# manter apenas as 5 superclasses
class_mask = (Y["label"].isin(classes)).values

X = X[class_mask]
Y = Y[class_mask].copy()

print("After superclass filter:")
print("X:", X.shape)
print("Y:", len(Y))

# =====================================================
# LABEL NUMÉRICO
# =====================================================

labels = Y["label"].map(class_to_idx).values

# =====================================================
# ONE HOT
# =====================================================

num_classes = 5

y_onehot = np.eye(num_classes)[labels]

print("One-hot labels:", y_onehot.shape)

# =====================================================
# SPLIT OFICIAL PTB-XL
# =====================================================

train_mask = (Y.strat_fold <= 8).values
val_mask = (Y.strat_fold == 9).values
test_mask = (Y.strat_fold == 10).values

X_train = X[train_mask]
X_val = X[val_mask]
X_test = X[test_mask]

y_train = y_onehot[train_mask]
y_val = y_onehot[val_mask]
y_test = y_onehot[test_mask]

# =====================================================
# VERIFICAÇÃO
# =====================================================

print("\nFinal split:")

print("x_train:", X_train.shape)
print("x_val  :", X_val.shape)
print("x_test :", X_test.shape)

print("y_train:", y_train.shape)
print("y_val  :", y_val.shape)
print("y_test :", y_test.shape)

# =====================================================
# SALVAR
# =====================================================

np.save(os.path.join(OUTPUT_DIR, "x_train.npy"), X_train.astype(np.float32))

np.save(os.path.join(OUTPUT_DIR, "x_val.npy"), X_val.astype(np.float32))

np.save(os.path.join(OUTPUT_DIR, "x_test.npy"), X_test.astype(np.float32))

np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train.astype(np.float32))

np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val.astype(np.float32))

np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test.astype(np.float32))

print("\nFiles saved to:")
print(OUTPUT_DIR)

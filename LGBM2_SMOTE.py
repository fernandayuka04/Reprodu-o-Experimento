import pandas as pd
import numpy as np
import os
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score

# --- 1. FÓRMULA DE INTERPOLAÇÃO (CONFORME IMAGEM 4 e 5) ---
def f_xi_interpolation(df):
    """Implementa a regra f(xi) para tratar valores ausentes."""
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        data = df_clean[col].values
        for i in range(len(data)):
            if np.isnan(data[i]):
                # Se xi é NaN e os vizinhos são válidos: f(xi) = (xi-1 + xi+1) / 2
                if i > 0 and i < len(data) - 1 and not np.isnan(data[i-1]) and not np.isnan(data[i+1]):
                    data[i] = (data[i-1] + data[i+1]) / 2
                else:
                    data[i] = 0 # Caso contrário f(xi) = 0
        df_clean[col] = data
    return df_clean

# --- 2. FUNÇÃO DE IMPRESSÃO DA TABELA ---
def print_detailed_results(title, y_test, y_pred):
    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, average=None, zero_division=0) * 100
    rec = recall_score(y_test, y_pred, average=None, zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, average=None, zero_division=0) * 100
    p_macro = precision_score(y_test, y_pred, average='macro', zero_division=0) * 100
    r_macro = recall_score(y_test, y_pred, average='macro', zero_division=0) * 100
    f_macro = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100

    print(f"\n--- {title} ---")
    print("Métrica | Classe 0 (Normal) | Classe 1 (Anômalo) | Média Macro")
    print("-" * 75)
    print(f"Acc.    | {acc:.2f}% | {acc:.2f}% | {acc:.2f}%")
    print(f"Prec.   | {prec[0]:.2f}% | {prec[1]:.2f}% | {p_macro:.2f}%")
    print(f"Recall  | {rec[0]:.2f}% | {rec[1]:.2f}% | {r_macro:.2f}%")
    print(f"F1-Score| {f1[0]:.2f}% | {f1[1]:.2f}% | {f_macro:.2f}%")

# --- 3. BLOCO DE EXECUÇÃO FORÇADA ---
print("Carregando dados...")
df_train = pd.read_csv('dataset_TREINO.csv')
df_test = pd.read_csv('dataset_TESTE.csv')

# Aplica fórmula de interpolação
df_train = f_xi_interpolation(df_train)
df_test = f_xi_interpolation(df_test)

# Filtra colunas numéricas
X_train = df_train.drop(columns=['FLAG']).select_dtypes(include=[np.number])
y_train = df_train['FLAG']
X_test = df_test[X_train.columns]
y_test = df_test['FLAG']

# Aplica SMOTE
print(f"Iniciando SMOTE e treinamento com {X_train.shape[1]} colunas...")
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# Configuração LGBM
model = LGBMClassifier(
    n_estimators=300, max_depth=30, learning_rate=0.2,
    num_leaves=70, min_child_samples=30, class_weight='balanced',
    objective='binary', random_state=42, verbose=-1
)

model.fit(X_res, y_res)
print_detailed_results("LGBM - RESULTADOS DETALHADOS (SMOTE)", y_test, model.predict(X_test))
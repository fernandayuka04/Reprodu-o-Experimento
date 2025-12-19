import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score


# --- 1. FÓRMULA DE INTERPOLAÇÃO (CONFORME IMAGEM 4 e 5) ---
def f_xi_interpolation(df):
    """
    Implementa a lógica matemática f(xi) para preenchimento de NaNs:
    - Se xi é NaN e os vizinhos não são: (xi-1 + xi+1) / 2
    - Se xi é NaN e algum vizinho é NaN: 0
    - Se xi não é NaN: mantém xi
    """
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        data = df_clean[col].values
        for i in range(len(data)):
            if np.isnan(data[i]):
                # Aplica a média se os vizinhos existem e não são NaN
                if i > 0 and i < len(data) - 1 and not np.isnan(data[i - 1]) and not np.isnan(data[i + 1]):
                    data[i] = (data[i - 1] + data[i + 1]) / 2
                else:
                    data[i] = 0  # Atribui 0 conforme a fórmula
        df_clean[col] = data
    return df_clean


# --- 2. FUNÇÃO PARA IMPRESSÃO DA TABELA DE RESULTADOS ---
def print_detailed_results(title, y_test, y_pred):
    """Gera a tabela de métricas detalhadas solicitada."""
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


# --- 3. EXECUÇÃO PRINCIPAL ---
print("Carregando e processando dados para Random Forest...")

# Verificação de existência dos arquivos
if os.path.exists('dataset_TREINO.csv') and os.path.exists('dataset_TESTE.csv'):
    # Carga e Interpolação
    df_train = f_xi_interpolation(pd.read_csv('dataset_TREINO.csv'))
    df_test = f_xi_interpolation(pd.read_csv('dataset_TESTE.csv'))

    # Preparação das Features (apenas numéricas)
    X_train = df_train.drop(columns=['FLAG']).select_dtypes(include=[np.number])
    y_train = df_train['FLAG']
    X_test = df_test[X_train.columns]
    y_test = df_test['FLAG']

    # Aplicação do SMOTE para igualar as classes
    print(f"Aplicando SMOTE em {X_train.shape[1]} colunas...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Configuração do Random Forest conforme Imagem 3:
    # n_estimators: 200, max_depth: 20, min_samples_split: 2, min_samples_leaf: 1
    model_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',  #
        random_state=42,
        n_jobs=-1
    )

    print("Treinando o modelo...")
    model_rf.fit(X_resampled, y_resampled)

    # Exibição dos resultados
    y_pred = model_rf.predict(X_test)
    print_detailed_results("RANDOM FOREST - RESULTADOS DETALHADOS (SMOTE)", y_test, y_pred)
else:
    print("Erro: Arquivos CSV não encontrados no diretório atual.")
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score
from scipy.interpolate import interp1d
import os

# ------------------------------------------------------------------------------
# 1. CONFIGURAÇÕES E CAMINHOS
# ------------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

# Nomes dos arquivos gerados na etapa anterior
FILE_TRAIN = 'dataset_treino_balanceado.csv'
FILE_VAL = 'dataset_validacao.csv'
FILE_TEST = 'dataset_treino_balanceado.csv'


# ------------------------------------------------------------------------------
# 2. FUNÇÕES DE SUPORTE (MANTIDAS DO SEU ORIGINAL)
# ------------------------------------------------------------------------------

def preprocess_data(df, fit_scaler=False, scaler=None):
    """
    Realiza interpolação linear, tratamento de outliers e normalização.
    Adaptada para receber o DataFrame e separar X e y automaticamente.
    """
    # Identifica colunas de ID e FLAG
    col_id = df.columns[0]
    col_flag = 'FLAG'

    y = df[col_flag].values
    # X contém apenas as colunas de consumo
    X_raw = df.drop(columns=[col_id, col_flag]).values

    # --- Interpolação (Fórmula da Imagem adaptada via interp1d) ---
    X_imputed = X_raw.copy()
    for i in range(X_imputed.shape[0]):
        series = X_imputed[i, :]
        not_nan_indices = np.where(~np.isnan(series))[0]
        nan_indices = np.where(np.isnan(series))[0]
        if len(not_nan_indices) >= 2:
            interp_func = interp1d(not_nan_indices, series[not_nan_indices], kind='linear', fill_value='extrapolate')
            series[nan_indices] = interp_func(nan_indices)
        series[np.isnan(series)] = 0
        X_imputed[i, :] = series

    # --- Tratamento de Outliers ---
    avg_x = np.mean(X_imputed, axis=0)
    std_x = np.std(X_imputed, axis=0)
    threshold_outlier = avg_x + 2 * std_x
    X_outlier_handled = X_imputed.copy()

    for j in range(X_outlier_handled.shape[1]):
        mask = X_outlier_handled[:, j] > threshold_outlier[j]
        X_outlier_handled[mask, j] = threshold_outlier[j]

    # --- Normalização ---
    if fit_scaler:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_outlier_handled)
        return X_scaled, y, scaler
    else:
        X_scaled = scaler.transform(X_outlier_handled)
        return X_scaled, y, scaler


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100

    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0) * 100
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0) * 100
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100

    prec_per_class = precision_score(y_test, y_pred, average=None, zero_division=0) * 100
    rec_per_class = recall_score(y_test, y_pred, average=None, zero_division=0) * 100
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0) * 100

    return {
        'Acc': accuracy,
        'Prec(avg)': prec_macro, 'Rec(avg)': rec_macro, 'F1(avg)': f1_macro,
        'Prec(0)': prec_per_class[0], 'Prec(1)': prec_per_class[1],
        'Rec(0)': rec_per_class[0], 'Rec(1)': rec_per_class[1],
        'F1(0)': f1_per_class[0], 'F1(1)': f1_per_class[1],
    }


def print_results(title, results):
    print(f"\n--- {title} ---")
    print("Métrica | Classe 0 (Anômalo) | Classe 1 (Normal) | Média Macro")
    print("-" * 65)
    acc = results['Acc']
    print(f"Acc.    | {acc:.2f}% | {acc:.2f}% | {acc:.2f}%")
    print(
        f"Prec.   | {results['Prec(0) Marc.'] if 'Prec(0) Marc.' in results else results['Prec(0)']:.2f}% | {results['Prec(1)']:.2f}% | {results['Prec(avg)']:.2f}%")
    print(f"Recall  | {results['Rec(0)']:.2f}% | {results['Rec(1)']:.2f}% | {results['Rec(avg)']:.2f}%")
    print(f"F1-Score| {results['F1(0)']:.2f}% | {results['F1(1)']:.2f}% | {results['F1(avg)']:.2f}%")
    print("-" * 65)


# ------------------------------------------------------------------------------
# 3. EXECUÇÃO PRINCIPAL
# ------------------------------------------------------------------------------

print("1. Carregando arquivos de Treino, Validação e Teste...")
try:
    df_train_raw = pd.read_csv(FILE_TRAIN)
    df_val_raw = pd.read_csv(FILE_VAL)
    df_test_raw = pd.read_csv(FILE_TEST)
except FileNotFoundError as e:
    print(f"Erro: Arquivos não encontrados. Verifique se {FILE_TRAIN} existe.")
    exit()

# 2. Pré-processamento
print("2. Aplicando Interpolação, Outliers e Scaler...")
X_train, y_train, scaler = preprocess_data(df_train_raw, fit_scaler=True)
X_val, y_val, _ = preprocess_data(df_val_raw, fit_scaler=False, scaler=scaler)
X_test, y_test, _ = preprocess_data(df_test_raw, fit_scaler=False, scaler=scaler)

print(f"Distribuição de Treino: {Counter(y_train)}")

# 3. Configuração do Modelo (Seus parâmetros otimizados)
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'n_estimators': 300,
    'learning_rate': 0.2,
    'num_leaves': 70,
    'max_depth': 30,
    'min_child_samples': 30,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'class_weight': 'balanced',  # Ajuste automático para o desequilíbrio
    'random_state': SEED,
    'n_jobs': -1
}

print("\n3. Treinando LGBM Classifier...")
model = LGBMClassifier(**LGBM_PARAMS)
# Usando o conjunto de validação para monitorar o treino
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='binary_logloss')

# 4. Avaliação Final
results = evaluate_model(model, X_test, y_test)
print_results("LGBM - RESULTADOS NO CONJUNTO DE TESTE", results)
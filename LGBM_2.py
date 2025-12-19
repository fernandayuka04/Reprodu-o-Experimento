import pandas as pd  # Manipulação de dados
import numpy as np  # Operações matemáticas
import os  # Caminhos de arquivos
from lightgbm import LGBMClassifier  # Modelo LightGBM
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score  # Métricas


def evaluate_model(model, X_test, y_test):
    """Calcula métricas detalhadas."""
    y_pred = model.predict(X_test)
    return {
        'Acc': accuracy_score(y_test, y_pred) * 100,
        'Prec': precision_score(y_test, y_pred, average=None, zero_division=0) * 100,
        'Rec': recall_score(y_test, y_pred, average=None, zero_division=0) * 100,
        'F1': f1_score(y_test, y_pred, average=None, zero_division=0) * 100,
        'Prec(avg)': precision_score(y_test, y_pred, average='macro', zero_division=0) * 100,
        'Rec(avg)': recall_score(y_test, y_pred, average='macro', zero_division=0) * 100,
        'F1(avg)': f1_score(y_test, y_pred, average='macro', zero_division=0) * 100
    }


def print_results(title, res):
    """Imprime a tabela de resultados."""
    print(f"\n--- {title} ---")
    print("Métrica | Classe 0 (Normal) | Classe 1 (Anômalo) | Média Macro")
    print("-" * 75)
    print(f"Acc.    | {res['Acc']:.2f}% | {res['Acc']:.2f}% | {res['Acc']:.2f}%")
    print(f"Prec.   | {res['Prec'][0]:.2f}% | {res['Prec'][1]:.2f}% | {res['Prec(avg)']:.2f}%")
    print(f"Recall  | {res['Rec'][0]:.2f}% | {res['Rec'][1]:.2f}% | {res['Rec(avg)']:.2f}%")
    print(f"F1-Score| {res['F1'][0]:.2f}% | {res['F1'][1]:.2f}% | {res['F1(avg)']:.2f}%")


# --- EXECUÇÃO ---
file_train = '../Métodos Supervisionados e SMOTE/dataset_TREINO.csv'
file_test = '../Métodos Supervisionados e SMOTE/dataset_TESTE.csv'

if os.path.exists(file_train) and os.path.exists(file_test):
    df_train = pd.read_csv(file_train)
    df_test = pd.read_csv(file_test)

    # LIMPEZA DE DADOS: Seleciona apenas colunas numéricas (exclui IDs em texto)
    # O modelo não pode receber strings como '82CB7B5...'
    X_train = df_train.drop(columns=['FLAG']).select_dtypes(include=[np.number])
    y_train = df_train['FLAG']

    # Garante que o Teste tenha as mesmas colunas que o Treino
    X_test = df_test[X_train.columns]
    y_test = df_test['FLAG']

    # Parâmetros das imagens enviadas
    LGBM_PARAMS = {
        'objective': 'binary', 'n_estimators': 300, 'max_depth': 30,
        'learning_rate': 0.2, 'num_leaves': 70, 'min_child_samples': 30,
        'class_weight': 'balanced', 'random_state': 42, 'verbose': -1
    }

    print(f"Iniciando treinamento com {X_train.shape[1]} colunas numéricas...")
    model_lgbm = LGBMClassifier(**LGBM_PARAMS)
    model_lgbm.fit(X_train, y_train)

    results = evaluate_model(model_lgbm, X_test, y_test)
    print_results("LGBM - RESULTADOS DETALHADOS", results)
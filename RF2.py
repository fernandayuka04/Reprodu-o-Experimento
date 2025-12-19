import pandas as pd # Importa a biblioteca para manipulação de dados
import numpy as np # Importa a biblioteca para operações matemáticas
import os # Importa a biblioteca para gerenciar caminhos de arquivos
from sklearn.ensemble import RandomForestClassifier # Importa o modelo Random Forest
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score # Métricas de avaliação

# --- FUNÇÕES DE APOIO ---

def evaluate_model(model, X_test, y_test):
    """Calcula métricas detalhadas de desempenho."""
    y_pred = model.predict(X_test) # O modelo realiza as previsões nos dados de teste
    results = {
        'Acc': accuracy_score(y_test, y_pred) * 100, # Calcula a acurácia global
        'Prec(avg)': precision_score(y_test, y_pred, average='macro', zero_division=0) * 100, # Precisão média
        'Rec(avg)': recall_score(y_test, y_pred, average='macro', zero_division=0) * 100, # Recall médio
        'F1(avg)': f1_score(y_test, y_pred, average='macro', zero_division=0) * 100, # F1 médio
        'Prec': precision_score(y_test, y_pred, average=None, zero_division=0) * 100, # Precisão por classe
        'Rec': recall_score(y_test, y_pred, average=None, zero_division=0) * 100, # Recall por classe
        'F1': f1_score(y_test, y_pred, average=None, zero_division=0) * 100 # F1 por classe
    }
    return results

def print_results(title, res):
    """Exibe os resultados formatados em tabela."""
    print(f"\n--- {title} ---")
    print("Métrica | Classe 0 (Normal) | Classe 1 (Anômalo) | Média Macro")
    print("-" * 75)
    print(f"Acc.    | {res['Acc']:.2f}% | {res['Acc']:.2f}% | {res['Acc']:.2f}%")
    print(f"Prec.   | {res['Prec'][0]:.2f}% | {res['Prec'][1]:.2f}% | {res['Prec(avg)']:.2f}%")
    print(f"Recall  | {res['Rec'][0]:.2f}% | {res['Rec'][1]:.2f}% | {res['Rec(avg)']:.2f}%")
    print(f"F1-Score| {res['F1'][0]:.2f}% | {res['F1'][1]:.2f}% | {res['F1(avg)']:.2f}%")

# --- EXECUÇÃO PRINCIPAL ---

# Definindo os arquivos gerados anteriormente
file_train = '../Métodos Supervisionados e SMOTE/dataset_TREINO.csv'
file_test = '../Métodos Supervisionados e SMOTE/dataset_TESTE.csv'

# Verifica se os arquivos existem antes de tentar carregar
if not os.path.exists(file_train) or not os.path.exists(file_test):
    print(f"ERRO: Arquivos não encontrados. Certifique-se de que {file_train} está na mesma pasta.")
else:
    print("Carregando dados para o Random Forest...")
    df_train = pd.read_csv(file_train) # Carrega o CSV de treino
    df_test = pd.read_csv(file_test)   # Carrega o CSV de teste

    # --- LIMPEZA E PREPARAÇÃO ---
    # 1. Removemos a coluna alvo (FLAG) para isolar as características (X)
    # 2. .select_dtypes(include=[np.number]) garante que apenas colunas numéricas sejam usadas
    # Isso evita o erro 'could not convert string to float'
    X_train = df_train.drop(columns=['FLAG']).select_dtypes(include=[np.number])
    y_train = df_train['FLAG'] # Define o rótulo: 0 para Normal, 1 para Anômalo

    # Garantimos que o conjunto de teste tenha exatamente as mesmas colunas numéricas que o treino
    X_test = df_test[X_train.columns]
    y_test = df_test['FLAG']

    # --- CONFIGURAÇÃO DO MODELO (CONFORME IMAGEM 3) ---
    RF_PARAMS = {
        'n_estimators': 200,          # Número de árvores na floresta
        'max_depth': 20,              # Profundidade máxima de cada árvore
        'min_samples_split': 2,       # Mínimo de amostras para dividir um nó
        'min_samples_leaf': 1,        # Mínimo de amostras em um nó folha
        'class_weight': 'balanced',   # Ajusta pesos para classes desbalanceadas
        'random_state': 42,           # Garante que o resultado seja sempre o mesmo
        'n_jobs': -1                  # Usa todos os processadores do seu computador
    }

    print(f"Iniciando treinamento do Random Forest com {X_train.shape[1]} colunas numéricas...")
    model_rf = RandomForestClassifier(**RF_PARAMS) # Cria o modelo com os parâmetros
    model_rf.fit(X_train, y_train) # O modelo aprende com os dados

    # --- AVALIAÇÃO ---
    print("Treinamento concluído. Avaliando resultados...")
    results_rf = evaluate_model(model_rf, X_test, y_test) # Testa o modelo com dados novos
    print_results("RANDOM FOREST - RESULTADOS DETALHADOS", results_rf) # Imprime a tabela
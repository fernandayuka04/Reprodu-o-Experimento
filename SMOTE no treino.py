import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

# 1. CARREGAMENTO
print("Carregando dados...")
df = pd.read_csv('dataset_treino.csv', low_memory=False)

# Detectar automaticamente o nome da coluna de ID (geralmente a primeira coluna)
# e garantir que a FLAG seja tratada corretamente.
coluna_id = df.columns[0]  # Assume que a primeira coluna é o ID (CONS_NO ou Customer_ID)
coluna_flag = 'FLAG'

print(f"Coluna de ID detectada: {coluna_id}")


# ---------------------------------------------------------
# FASE 1: TRATAMENTO DE NaNs (FÓRMULA DA IMAGEM)
# ---------------------------------------------------------

def aplicar_formula_limpeza(row):
    # Selecionamos apenas os valores de consumo (excluindo ID e FLAG)
    # Usamos uma máscara booleana para evitar erro de nome de coluna
    colunas_para_ignorar = [coluna_id, coluna_flag]
    colunas_consumo = [c for c in row.index if c not in colunas_para_ignorar]

    consumo = row[colunas_consumo].values.astype(float)
    novo_consumo = consumo.copy()

    # Loop para aplicar a fórmula da imagem: x_i = (x_i-1 + x_i+1) / 2
    for i in range(1, len(consumo) - 1):
        if np.isnan(consumo[i]):
            # Regra 1: Média se os vizinhos forem números válidos
            if not np.isnan(consumo[i - 1]) and not np.isnan(consumo[i + 1]):
                novo_consumo[i] = (consumo[i - 1] + consumo[i + 1]) / 2
            # Regra 2: Vira 0 se algum vizinho for NaN
            else:
                novo_consumo[i] = 0

    # Tratar a primeira e última posição (bordas)
    novo_consumo[0] = 0 if np.isnan(novo_consumo[0]) else novo_consumo[0]
    novo_consumo[-1] = 0 if np.isnan(novo_consumo[-1]) else novo_consumo[-1]

    # Atualiza a linha com os valores limpos
    row[colunas_consumo] = novo_consumo
    return row


print("Limpando dados com a fórmula da imagem... (Aguarde)")
# axis=1 aplica a função em cada linha (cliente por cliente)
df_limpo = df.apply(aplicar_formula_limpeza, axis=1)

# ---------------------------------------------------------
# FASE 2: PREPARAÇÃO E SMOTE
# ---------------------------------------------------------

# X recebe apenas as colunas de consumo (dados numéricos para o SMOTE)
X = df_limpo.drop(columns=[coluna_id, coluna_flag])
# y recebe o alvo (quem é normal e quem é anômalo)
y = df_limpo[coluna_flag]

print(f"\nDistribuição antes do SMOTE: {dict(y.value_counts())}")

# Aplicando o SMOTE para igualar as quantidades
# sampling_strategy='auto' iguala a classe minoritária à majoritária
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print(f"Distribuição após o SMOTE: {dict(pd.Series(y_res).value_counts())}")

# ---------------------------------------------------------
# FASE 3: SALVAMENTO
# ---------------------------------------------------------

# Recriamos o DataFrame final juntando os novos dados X e y
df_balanceado = pd.DataFrame(X_res, columns=X.columns)
df_balanceado[coluna_flag] = y_res

# Salvamos para usar no treinamento do modelo supervisionado
df_balanceado.to_csv('dataset_treino_balanceado.csv', index=False)
print("\nArquivo 'dataset_treino_balanceado.csv' gerado com sucesso!")
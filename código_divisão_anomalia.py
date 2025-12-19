import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Carregar o dataset (que já contém apenas anômalos)
INPUT_FILE = 'clientes_anomalos_SGCC.csv'
print("Carregando dados de anomalias...")
df = pd.read_csv(INPUT_FILE, low_memory=False)

# 2. Definir X e y
# Mesmo que todos sejam FLAG=0 (ou 1), precisamos separar para o split
y = df['FLAG']
X = df.drop(columns=['FLAG'])

# ----------------------------------------------------------------
# DIVISÃO DOS DADOS (Sem estratificação, pois só há uma classe)
# ----------------------------------------------------------------

# Passo 1: 70% Treino, 30% Resto
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42  # Mantém a reprodutibilidade
)

# Passo 2: Divide os 30% restantes em Validação (15%) e Teste (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42
)

# ----------------------------------------------------------------
# FUNÇÃO PARA SALVAR
# ----------------------------------------------------------------

def salvar_dataset(X_part, y_part, nome):
    final_df = X_part.copy()
    final_df['FLAG'] = y_part
    file_name = f'anomalias_{nome}.csv'
    final_df.to_csv(file_name, index=False)
    print(f"Arquivo {file_name} salvo! Conteúdo: {len(final_df)} registros.")
    return final_df

print("\nProcessando divisões...")
df_train = salvar_dataset(X_train, y_train, 'treino')
df_val = salvar_dataset(X_val, y_val, 'validacao')
df_test = salvar_dataset(X_test, y_test, 'teste')

# ----------------------------------------------------------------
# RESUMO FINAL
# ----------------------------------------------------------------

print("\n" + "=" * 40)
print("RESUMO DA DIVISÃO DE ANOMALIAS")
print("=" * 40)
print(f"Total Original: {len(df)}")
print(f"Treino:     {len(df_train)} (70%)")
print(f"Validação:  {len(df_val)} (15%)")
print(f"Teste:      {len(df_test)} (15%)")
import pandas as pd

# Carregue o arquivo
df = pd.read_csv('data.csv', low_memory=False)

# Peça para o Python contar a quantidade real de linhas e colunas
print(f"Total de Clientes (Linhas): {df.shape[0]}")
print(f"Total de Informações (Colunas): {df.shape[1]}")

# Verifique se há clientes com FLAG=0 e FLAG=1
print("\nContagem por tipo de cliente:")
print(df['FLAG'].value_counts())
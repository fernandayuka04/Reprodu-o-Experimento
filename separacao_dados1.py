import pandas as pd

# 1. Carregar o arquivo de dados completo
# Observação: Se o arquivo for muito grande (mais de 1GB), use 'low_memory=False'
file_path = 'data.csv'
df = pd.read_csv(file_path)

# Renomear colunas para facilitar a compreensão, se necessário
# df.rename(columns={'CONS_NO': 'Customer_ID'}, inplace=True)

print(f"Total de Registros Carregados: {len(df)}")
print(f"Colunas disponíveis: {df.columns.tolist()}")

# 2. Divisão da Base de Dados usando a coluna 'FLAG'
# Onde 'FLAG' é igual a 1 (Anômalo)
df_anomalous = df[df['FLAG'] == 1].copy()

# Onde 'FLAG' é igual a 0 (Normal)
df_normal = df[df['FLAG'] == 0].copy()

# 3. Exibir a Contagem para Confirmação
print("-" * 30)
print(f"Clientes Anômalos (FLAG = 1): {len(df_anomalous)} linhas")
print(f"Clientes Normais (FLAG = 0): {len(df_normal)} linhas")
print(f"Total Verificado: {len(df_anomalous) + len(df_normal)} linhas")
print("-" * 30)

# 4. Salvar os novos arquivos CSV
# Salvando o arquivo com os clientes anômalos
df_anomalous.to_csv('clientes_anomalos_SGCC.csv', index=False)

# Salvando o arquivo com os clientes normais
df_normal.to_csv('clientes_normais_SGCC.csv', index=False)

print("Arquivos salvos com sucesso!")
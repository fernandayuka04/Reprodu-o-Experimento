import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Carregar o dataset original
# Certifique-se de que o arquivo 'data.csv' está na mesma pasta
INPUT_FILE = 'data set.csv'
print("Carregando dados...")
df = pd.read_csv(INPUT_FILE, low_memory=False)

# 2. Definir as variáveis de destino (Target)
# Queremos dividir os CLIENTES, mantendo a proporção da coluna FLAG
y = df['FLAG']
X = df.drop(columns=['FLAG'])  # Mantemos o Customer_ID e as datas

# ----------------------------------------------------------------
# ESTRATÉGIA: DIVISÃO ESTRATIFICADA POR CLIENTE
# ----------------------------------------------------------------

# Passo 1: Separar 70% para TREINO e 30% para o RESTO (Validação + Teste)
# O parâmetro 'stratify=y' garante que a proporção de FLAG=0 e FLAG=1 seja mantida
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

# Passo 2: Dos 30% restantes, dividir meio a meio para VALIDAÇÃO e TESTE
# Isso resulta em 15% para cada um em relação ao total original
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)


# ----------------------------------------------------------------
# FASE DE SALVAMENTO: REUNIR X E Y NOVAMENTE
# ----------------------------------------------------------------

def salvar_dataset(X_part, y_part, nome):
    # Junta as colunas de dados com a coluna FLAG novamente
    final_df = X_part.copy()
    final_df['FLAG'] = y_part

    # Salva o arquivo final
    file_name = f'dataset_{nome}.csv'
    final_df.to_csv(file_name, index=False)
    print(f"Arquivo {file_name} salvo com sucesso! ({len(final_df)} clientes)")
    return final_df


print("\nSalvando os arquivos...")
df_train = salvar_dataset(X_train, y_train, 'treino')
df_val = salvar_dataset(X_val, y_val, 'validacao')
df_test = salvar_dataset(X_test, y_test, 'teste')

# ----------------------------------------------------------------
# RESUMO DA DIVISÃO
# ----------------------------------------------------------------

print("\n" + "=" * 40)
print("RESUMO DA DISTRIBUIÇÃO (ESTRATIFICADA)")
print("=" * 40)

for nome, d in [("TREINO", df_train), ("VALIDAÇÃO", df_val), ("TESTE", df_test)]:
    total = len(d)
    anomalos = len(d[d['FLAG'] == 0])  # Conforme seu critério FLAG=0
    normais = len(d[d['FLAG'] == 1])
    print(f"{nome}: {total} clientes | Anômalos: {anomalos} ({anomalos / total:.1%}) | Normais: {normais}")
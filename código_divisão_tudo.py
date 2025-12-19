import pandas as pd
from sklearn.model_selection import train_test_split

# Configurações de arquivos
FILE_ANOMALOS = 'clientes_anomalos_SGCC.csv'
FILE_NORMAIS = 'clientes_normais_SGCC.csv'


def processar_datasets():
    # 1. Carregar Dados
    print("Carregando datasets...")
    df_anom = pd.read_csv(FILE_ANOMALOS, low_memory=False)
    df_norm = pd.read_csv(FILE_NORMAIS, low_memory=False)

    # 2. Atribuir as Flags conforme solicitado
    df_anom['FLAG'] = 1  # 1 = Anômalo
    df_norm['FLAG'] = 0  # 0 = Normal

    # 3. Função interna para dividir os dados de cada classe
    def dividir(df):
        # 70% Treino, 30% Temporário
        train, temp = train_test_split(df, test_size=0.30, random_state=42)
        # 15% Validação, 15% Teste
        val, test = train_test_split(temp, test_size=0.50, random_state=42)
        return train, val, test

    # Dividindo anômalos e normais separadamente para manter a proporção
    train_anom, val_anom, test_anom = dividir(df_anom)
    train_norm, val_norm, test_norm = dividir(df_norm)

    # 4. Juntar e Embaralhar (Shuffle)
    # Criamos os 3 conjuntos finais misturando as classes
    df_treino = pd.concat([train_anom, train_norm]).sample(frac=1, random_state=42)
    df_val = pd.concat([val_anom, val_norm]).sample(frac=1, random_state=42)
    df_teste = pd.concat([test_anom, test_norm]).sample(frac=1, random_state=42)

    # 5. Opcional: Criar um arquivo único com identificação da divisão
    # Útil se você quiser manter tudo em um CSV só, mas rotulado
    df_treino['DIVISAO'] = 'TREINO'
    df_val['DIVISAO'] = 'VALIDACAO'
    df_teste['DIVISAO'] = 'TESTE'

    df_completo_unificado = pd.concat([df_treino, df_val, df_teste])

    # 6. Salvar Arquivos
    print("\nSalvando arquivos finais...")

    # Salva os 3 separados (recomendado para o modelo)
    df_treino.drop(columns=['DIVISAO']).to_csv('dataset_TREINO.csv', index=False)
    df_val.drop(columns=['DIVISAO']).to_csv('dataset_VALIDACAO.csv', index=False)
    df_teste.drop(columns=['DIVISAO']).to_csv('dataset_TESTE.csv', index=False)

    # Salva o unificado (como você pediu)
    df_completo_unificado.to_csv('dataset_COMPLETO_SUPERVISIONADO.csv', index=False)

    # Resumo para conferência
    print("-" * 50)
    print("RESUMO DA COMPOSIÇÃO:")
    for nome, d in [("TREINO", df_treino), ("VALIDAÇÃO", df_val), ("TESTE", df_teste)]:
        total = len(d)
        anom = len(d[d['FLAG'] == 1])
        norm = len(d[d['FLAG'] == 0])
        print(f"{nome}: {total} total | {anom} Anômalos (1) | {norm} Normais (0)")
    print("-" * 50)


if __name__ == "__main__":
    processar_datasets()
### MACHINE LEARNING ### 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Preparação dos Dados para Machine Learning ---

print("\n--- Iniciando Pré-processamento para ML ---")

# 1. Seleção de Features (Colunas) e Criação da Variável Alvo (Target)
# usar um subconjunto do DataFrame original para simplificar
features = ['Age', 'Height', 'Weight', 'Sex', 'NOC', 'Sport']
df_ml = df[features + ['Medal']].copy() # .copy() para evitar warnings

# Criar a variável alvo 'Medal_Won' (1 para quem ganhou medalha, 0 para quem não ganhou)
df_ml['Medal_Won'] = df_ml['Medal'].notna().astype(int)
df_ml.drop('Medal', axis=1, inplace=True)


# 2. Tratamento de Valores Faltantes (NaN)
# Preencher NaN em colunas numéricas com a mediana é uma abordagem robusta
for col in ['Age', 'Height', 'Weight']:
    median_value = df_ml[col].median()
    df_ml[col].fillna(median_value, inplace=True)
    print(f"Valores nulos em '{col}' preenchidos com a mediana: {median_value}")


# 3. Conversão de Variáveis Categóricas em Numéricas (One-Hot Encoding)
# Modelos de ML não entendem texto, então transformamos 'Sex', 'NOC' e 'Sport' em colunas de 0s e 1s
df_ml_encoded = pd.get_dummies(df_ml, columns=['Sex', 'NOC', 'Sport'], drop_first=True)
print("\nDataFrame transformado com One-Hot Encoding.")


# 4. Separar Features (X) e Alvo (y)
X = df_ml_encoded.drop('Medal_Won', axis=1)
y = df_ml_encoded['Medal_Won']


# 5. Dividir os dados em Conjunto de Treino e Conjunto de Teste
# 80% dos dados para treino, 20% para teste. random_state para garantir que a divisão seja sempre a mesma.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# 'stratify=y' é importante aqui para garantir que a proporção de medalhistas
# seja a mesma nos datasets de treino e teste.

print("\n--- Dados Prontos para Treinamento ---")
print(f"Formato de X_train (features de treino): {X_train.shape}")
print(f"Formato de X_test (features de teste): {X_test.shape}")
print(f"Formato de y_train (alvo de treino): {y_train.shape}")
print(f"Formato de y_test (alvo de teste): {y_test.shape}")
print(f"\nProporção de medalhistas no dataset de treino: {y_train.mean():.4f}")
print(f"Proporção de medalhistas no dataset de teste: {y_test.mean():.4f}")
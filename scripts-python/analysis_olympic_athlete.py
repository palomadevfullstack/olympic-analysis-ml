# importar bibliotecas
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

# configurações para melhorar a visualização 
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# carregar dataset
try:
    df = pd.read_csv('athlete_events.csv')
except FileNotFoundError:
    print("Erro: Arquivo 'athlete_events.csv' não encontrado. Verifique o caminho.")
    exit()
    
    
# Primeiras inspeções 

# 1. Visualizar as primeiras 5 linhas
print("--- Amostra dos Dados ---")
df.head()
print("\n")

# 2. Informações sobre as colunas, tipos e valores nulos
print("--- Informações do DataFrame ---")
df.info()
print("\n")

# 3. Estatísticas descritivas das colunas numéricas
print("--- Estatísticas Descritivas ---")
print(df.describe())

# ---- Análise 1: Evolução de Perfil Físico em Natação vs Ginástica Artística ----

# 1. Filtrar os dados para os esportes de interesse e apenas medalhistas
sports_of_interest = ['Swimming', 'Artistic Gymnastics']
df_sports = df[df['Sport'].isin(sports_of_interest)]
df_medals = df_sports.dropna(subset=['Medal']) # Pega apenas linhas onde a medalha não é nula

print(f"\nTotal de medalhistas em Natação e Ginástica Artística: {len(df_medals)}")

# 2. Remover linhas onde altura ou peso são nulos para esta análise
df_physical = df_medals.dropna(subset=['Height', 'Weight'])
print(f"Total de medalhistas com dados físicos: {len(df_physical)}")

# 3. Agrupar por Ano, Esporte e Sexo e calcular a média de Altura e Peso
evolution= df_physical.groupby(['Year', 'Sport', 'Sex'])[['Height', 'Weight']].mean().reset_index()

# Arredondar para 2 casas decimais para melhor visualização 
evolution[['Height', 'Weight']] = evolution[['Height', 'Weight']].round(2)

# 4. Visualizar os resultados

# Gráfico da Evolução da Altura Média
plt.figure(figsize=(15, 7))
sns.lineplot(data=evolution, x='Year', y='Height', hue='Sport', style='Sex', lw=2.5)
plt.title('Evolução da Altura Média de Medalhistas Olímpicos (Natação vs. Ginástica)', fontsize=16)
plt.xlabel('Ano', fontsize=12)
plt.ylabel('Altura Média (cm)', fontsize=12)
plt.legend(title='Esporte / Sexo')
plt.grid(True)
plt.show()

# Gráfico da Evolução do Peso Médio
plt.figure(figsize=(15, 7))
sns.lineplot(data=evolution, x='Year', y='Weight', hue='Sport', style='Sex', lw=2.5)
plt.title('Evolução do Peso Médio de Medalhistas Olímpicos (Natação vs. Ginástica)', fontsize=16)
plt.xlabel('Ano', fontsize=12)
plt.ylabel('Peso Médio (kg)', fontsize=12)
plt.legend(title='Esporte / Sexo')
plt.grid(True)
plt.show()

# --- Análise 2: Potências Olímpicas e Especialistas por Esporte ---

# 1. Obter um ranking geral de medalhas por país (Comitê Olímpico Nacional - NOC)
print("\n--- Top 15 Países por Número Total de Medalhas ---")
medalhas_por_pais = df.dropna(subset=['Medal'])['NOC'].value_counts().head(15)
print(medalhas_por_pais)

# Visualizar este ranking
plt.figure(figsize=(15, 8))
sns.barplot(x=medalhas_por_pais.index, y=medalhas_por_pais.values, palette='viridis')
plt.title('Top 15 Países por Número Total de Medalhas Olímpicas', fontsize=16)
plt.xlabel('País (Comitê Olímpico Nacional)', fontsize=12)
plt.ylabel('Número Total de Medalhas', fontsize=12)
plt.xticks(rotation=45)
plt.show()


# 2. Criar uma "Matriz de Força" para ver a distribuição de medalhas
# Usar uma pivot table para cruzar os países (NOC) com os esportes
df_medalhas = df.dropna(subset=['Medal'])

# Pegar a lista dos top 10 países para manter o gráfico legível
top_10_paises = medalhas_por_pais.head(10).index

# Filtrar o DataFrame para conter apenas esses países
df_top_10 = df_medalhas[df_medalhas['NOC'].isin(top_10_paises)]

# Criar a pivot table: Países nas linhas, Esportes nas colunas, contagem de medalhas como valores
matriz_forca = df_top_10.pivot_table(
    index='NOC', 
    columns='Sport', 
    values='Medal', 
    aggfunc='count'
).fillna(0)


# 3. Visualizar a Matriz de Força com um Heatmap
plt.figure(figsize=(20, 12))
sns.heatmap(
    matriz_forca, 
    cmap='rocket_r', # Usando um mapa de cores invertido (cores mais escuras = mais medalhas)
    annot=True,     # anota o número de medalhas em cada célula
    fmt='.0f',      # formata a anotação como um número inteiro
    linewidths=.5
)
plt.title('Distribuição de Medalhas por Esporte para os Top 10 Países', fontsize=18)
plt.xlabel('Esporte', fontsize=14)
plt.ylabel('País', fontsize=14)
plt.show()

# --- Análise 3: Deep Dive no Brasil (BRA) ---

print("\n--- Iniciando Análise Focada no Brasil ---")

# 1. Filtrar o DataFrame para conter apenas medalhas do Brasil
df_brazil = df[df['NOC'] == 'BRA'].dropna(subset=['Medal'])

# Checar se o filtro funcionou
print(f"Total de medalhas olímpicas do Brasil no dataset: {len(df_brazil)}")


# 2. Análise dos Esportes mais vitoriosos para o Brasil
top_sports_br = df_brazil['Sport'].value_counts()

plt.figure(figsize=(15, 8))
sns.barplot(x=top_sports_br.index, y=top_sports_br.values, palette='Greens_r')
plt.title('Distribuição de Medalhas por Esporte para o Brasil', fontsize=16)
plt.xlabel('Esporte', fontsize=12)
plt.ylabel('Número de Medalhas', fontsize=12)
plt.xticks(rotation=75) # Rotacionar os nomes dos esportes para melhor leitura
plt.show()


# 3. Análise da Evolução de Medalhas do Brasil por Ano
# Vamos contar as medalhas por ano de edição dos Jogos
# Usamos .unique() nos 'Games' para não contar medalhas de equipe múltiplas vezes por evento
# Ex: Ouro no Vôlei conta como 1, não como 12.
medalhas_por_ano = df_brazil.groupby('Year')['Event'].nunique().reset_index()

plt.figure(figsize=(15, 8))
sns.lineplot(data=medalhas_por_ano, x='Year', y='Event', lw=3, marker='o', color='green')
plt.title('Evolução do Número de Medalhas do Brasil por Edição Olímpica', fontsize=16)
plt.xlabel('Ano', fontsize=12)
plt.ylabel('Número de Medalhas Conquistadas', fontsize=12)

# Adicionar uma anotação para destacar as Olimpíadas do Rio 2016
ano_rio = 2016
medalhas_rio = medalhas_por_ano[medalhas_por_ano['Year'] == ano_rio]['Event'].iloc[0]
plt.annotate(
    f'Rio 2016 ({medalhas_rio} medalhas)',
    xy=(ano_rio, medalhas_rio),
    xytext=(ano_rio - 20, medalhas_rio + 1),
    arrowprops=dict(facecolor='black', shrink=0.05),
    fontsize=12,
    fontweight='bold'
)
plt.grid(True)
plt.show()


# 4. Distribuição entre Ouro, Prata e Bronze
medal_distribution_br = df_brazil['Medal'].value_counts()

plt.figure(figsize=(10, 8))
plt.pie(
    medal_distribution_br,
    labels=medal_distribution_br.index,
    autopct='%1.1f%%', # Formata para mostrar a porcentagem
    colors=['#FFD700', '#C0C0C0', '#CD7F32'], # Cores para Ouro, Prata e Bronze
    startangle=140,
    textprops={'fontsize': 14, 'fontweight': 'bold'}
)
plt.title('Distribuição de Medalhas do Brasil (Ouro, Prata, Bronze)', fontsize=16)
plt.ylabel('') # Remove o label 'Medal' que o pie chart adiciona por padrão
plt.show()





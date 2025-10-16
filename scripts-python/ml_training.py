from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# --- Treinamento do Modelo de Machine Learning ---

print("\n--- Iniciando Treinamento do Modelo ---")

# 1. Inicializar o modelo
# Usamos max_iter=1000 para garantir que o modelo tenha iterações suficientes para convergir.
model = LogisticRegression(max_iter=1000)


# 2. Treinar o modelo com os dados de treino
print("Treinando o modelo de Regressão Logística...")
model.fit(X_train, y_train)
print("Treinamento concluído!")


# 3. Fazer previsões com os dados de teste
# O modelo nunca viu esses dados antes, então é um teste justo de sua performance.
y_pred = model.predict(X_test)


# 4. Avaliar a performance do modelo
print("\n--- Avaliação do Modelo nos Dados de Teste ---")
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia Geral: {accuracy:.4f}")

# O Classification Report é a melhor forma de avaliar um modelo de classificação.
# Ele dá a Precisão, Recall e F1-Score para cada classe.
print("\nRelatório de Classificação Detalhado:")
print(classification_report(y_test, y_pred, target_names=['Não Ganhou Medalha (0)', 'Ganhou Medalha (1)']))

## -- Treinamento avançado para melhorar acurácia utilizando o Random Forest 
from sklearn.ensemble import RandomForestClassifier

# --- Treinamento do Modelo Avançado: Random Forest ---

print("\n\n--- Iniciando Treinamento do Modelo Avançado: Random Forest ---")

# 1. Inicializar o modelo
# n_jobs=-1 usa todos os processadores do computador para acelerar o treinamento.
# random_state=42 garante que o resultado seja o mesmo toda vez que rodarmos.
rf_model = RandomForestClassifier(n_jobs=-1, random_state=42)


# 2. Treinar o modelo
# Este passo pode demorar um pouco mais que o anterior.
print("Treinando o modelo Random Forest... (isso pode levar alguns minutos)")
rf_model.fit(X_train, y_train)
print("Treinamento do Random Forest concluído!")


# 3. Fazer previsões com os dados de teste
y_pred_rf = rf_model.predict(X_test)


# 4. Avaliar a performance do novo modelo
print("\n--- Avaliação do Modelo Random Forest nos Dados de Teste ---")
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Acurácia Geral (Random Forest): {accuracy_rf:.4f}")

print("\nRelatório de Classificação Detalhado (Random Forest):")
print(classification_report(y_test, y_pred_rf, target_names=['Não Ganhou Medalha (0)', 'Ganhou Medalha (1)']))

# --- Etapa Final: Interpretando o Modelo Random Forest ---

print("\n\n--- Interpretando o Modelo: Features Mais Importantes ---")

# 1. Extrair a importância de cada feature do modelo treinado
importances = rf_model.feature_importances_
feature_names = X_train.columns

# 2. Criar um DataFrame para facilitar a visualização
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# 3. Ordenar o DataFrame pela importância
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)


# 4. Visualizar as 20 features mais importantes
plt.figure(figsize=(15, 10))
sns.barplot(
    x='Importance', 
    y='Feature', 
    data=feature_importance_df.head(20), # Pegando as top 20
    palette='summer'
)
plt.title('Top 20 Features Mais Importantes para Prever uma Medalha', fontsize=16)
plt.xlabel('Importância Relativa', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.show()
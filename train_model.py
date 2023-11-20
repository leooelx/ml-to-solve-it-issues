import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib

# Carregar o conjunto de dados a partir do JSON
with open('dataset.json', 'r') as file:
    dataset = json.load(file)

# Extrair descrições e soluções do conjunto de dados
descriptions = [item['descricao_problema'] for item in dataset]
solutions = [item['solucao_esperada'] for item in dataset]

# Vetorizar as descrições usando TF-IDF
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(descriptions)

# Inicializar e treinar um modelo LinearSVC
model = LinearSVC()
model.fit(X_vectorized, solutions)

# Salvar o modelo e o vetorizador para uso futuro
joblib.dump(model, 'modelo_resposta_problema.joblib')
joblib.dump(vectorizer, 'vetorizador_problemas.joblib')

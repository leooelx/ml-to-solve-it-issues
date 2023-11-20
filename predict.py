import joblib

# Carregar o modelo treinado e o vetorizador
modelo = joblib.load('modelo_resposta_problema.joblib')
vetorizador = joblib.load('vetorizador_problemas.joblib')

# Função para prever a solução com base em uma nova descrição do problema
def prever_solucao(nova_descricao):
    nova_descricao_vectorizada = vetorizador.transform([nova_descricao])
    solucao_prevista = modelo.predict(nova_descricao_vectorizada)[0]
    return solucao_prevista

# Exemplo de uso
nova_descricao_problema = input("Digite a descrição do problema: ")
solucao_prevista = prever_solucao(nova_descricao_problema)

print(f"Solução Prevista: {solucao_prevista}")

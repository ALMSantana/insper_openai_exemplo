# Importando bibliotecas necessárias
import openai  # pip install openai
import pandas as pd  # pip install pandas
from dotenv import load_dotenv  # pip install python-dotenv
import os  # faz parte da biblioteca padrão, não é necessário instalar
from io import StringIO  # faz parte da biblioteca padrão, não é necessário instalar

def openai_gerar_df_sentimentos(entrada, openai):
    # Informa ao usuário que a análise com OpenAI está em progresso
    print("Processando com OpenAI ...")

    # Define o prompt do sistema para instruir o modelo de linguagem
    prompt_sistema = """
    Faça uma análise de sentimentos dos comentários abaixo e apresente os resultados da seguinte forma, após avaliar cada comentário:

    1) Categoria mais forte associada ao comentário
    2) Classificação entre positivo, neutro ou negativo
    3) Ponto focal de melhoria ou de valoração indicado no comentário

    Após isso, compile também isso no formato de uma tabela (csv), com delimitador ;, para ser lido com a biblitoeca pandas.
    """

    # Define o prompt do usuário incluindo a entrada do CSV
    prompt_usuario =f'Aqui está o arquivo csv "{entrada}". Gere uma saída no formato CSV com cabeçalho Ocorrência;Categoria;Classificação;Ponto_focal;Comentário. Conteúdos vazios devem ser deixados em branco.'

    # Solicita uma resposta do modelo de linguagem da OpenAI
    resposta = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {
                "role": "system",
                "content" : prompt_sistema
            },
            {
                "role": "user",
                "content": prompt_usuario
            }
        ],
        temperature = 0 # Configura o modelo para ter respostas mais determinísticas
    )

    # Converte a resposta em formato CSV para um dataframe do pandas
    resultado = resposta["choices"][0]["message"]["content"]
    # Usando o delimitador ';' para ler o CSV e especificando o engine para python
    df = pd.read_csv(StringIO(resultado), sep=';', engine='python')
    df.to_csv("Dados.csv", sep=';', index=False)
    return df

def main():
    # Carrega variáveis de ambiente
    load_dotenv()
    api_openai = os.getenv("API_KEY_OPENAI")
    openai.api_key = api_openai

    # Lê os comentários do arquivo CSV
    df = pd.read_csv("olist_order_reviews_dataset.csv")
    dados = "".join(df['review_comment_message'].dropna().to_list())

    # Realiza a análise de sentimentos com a OpenAI e exibe o resultado
    teste = openai_gerar_df_sentimentos(dados, openai)
    print(teste)

# Se o script for executado como principal, chama a função main
if __name__ == "__main__":
    main()

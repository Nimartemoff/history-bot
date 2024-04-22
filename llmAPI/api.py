from g4f.client import Client
from g4f.Provider import RetryProvider, Liaobots, Chatgpt4Online, Vercel, GptForLove, GptTalkRu, Koala, FlowGpt
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import chromadb

llm = Client(provider=RetryProvider([Koala, FlowGpt, Liaobots, GptTalkRu, Chatgpt4Online, GptForLove, Vercel],
                                        shuffle=False))

embed_model = LangchainEmbedding(HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        # model_kwargs={"device": "cuda"},
        # encode_kwargs={"device": "cuda", "batch_size": 128},
        cache_folder='./local/model'
))


chroma_client = chromadb.PersistentClient()
chroma_collection = chroma_client.get_collection('quickstart')


def get_answer(question):
    question_embedding = embed_model.get_query_embedding(question)

    chunks = chroma_collection.query(query_embeddings=[question_embedding], n_results=5)

    context = ""

    for meta, doc in zip(chunks['metadatas'][0], chunks['documents'][0]):
        chunk = f"{doc}\n[Источник: {meta['file_name']} (стр. {meta['page_label']})]\n"
        context += chunk

    prompt = f"""
        Учитывая информацию только из предоставленного контекста и никаких своих знаний, ответь на вопрос.
        Если в контексте нет ответа на вопрос, то ответь, что не знаешь.
        Отвечай подробно, когда в контексте есть ответ.
        Обязательно указывай источник со страницей.
        Далее предоставлен контекст с источниками:
        ---------------------
        {context}
        ---------------------
        Вопрос:
        ---------------------
        {question}
        ---------------------
        """

    response = llm.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    
    return response.choices[0].message.content
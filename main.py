from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from  langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
import  os
from dotenv import load_dotenv


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

model=ChatOpenAI(
    model="gpt-4o-mini",
    api_key=api_key,
    temperature=0.5
)

arquivo=PyPDFLoader(r"Arquitetura RAG\5564-langchain\Projeto1\regras_futebol.pdf").load()

#chunk_size=1000, chunk_overlap=100
pedacos=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100
                                       ).split_documents(arquivo)

embeddings=OpenAIEmbeddings()

#FAISS
vetor=FAISS.from_documents(pedacos, embeddings)

while True:
    
    pergunta=input("Digite sua pergunta: ")
    if pergunta.lower() == "sair":
        break
    trechos=vetor.similarity_search(pergunta, k=2)
    contexto="\n\n".join([t.page_content for t in trechos])
    prompt=ChatPromptTemplate.from_messages([
    ("system", "Responda usando exclusivamente o conteudo fornecido:"),
    ("user", f"{pergunta} \n\n Contexto: {contexto}")
])
    cadeia=prompt | model    | StrOutputParser()
    resposta=cadeia.invoke({"query": pergunta, "context": contexto})
    print(resposta)
import os
from dotenv import load_dotenv
import streamlit as st

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

# ✅ Load env
load_dotenv()
st.title("Customer Support")
with st.sidebar:
    st.title("provide your api key")
    openai_api_key = st.text_input("Azure OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please enter your Azure OpenAI API key in the sidebar.")
    st.stop()

azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = "https://venug-mpf9rds3-koreacentral.cognitiveservices.azure.com/"
api_version = "2024-12-01-preview"

# ✅ LLM
llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1",
    api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
    temperature=0.7
)

# ✅ Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
    model="text-embedding-3-small"
)

# ✅ Load docs
loader = TextLoader("product-data.txt")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# ✅ Vector DB
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# ---------------------------------------------------
# ✅ STEP 1: Reformulate question using history
# ---------------------------------------------------

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the question using chat history. Do NOT answer."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

rewrite_chain = rewrite_prompt | llm | StrOutputParser()

# ---------------------------------------------------
# ✅ STEP 2: Final QA prompt
# ---------------------------------------------------

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant.
Use the context to answer.
Rephrase, don't copy paste.
If unknown, say you don't know.
Max 3 sentences.

Context:
{context}
"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_chain = qa_prompt | llm | StrOutputParser()

# ---------------------------------------------------
# ✅ STEP 3: Full pipeline (manual history-aware RAG)
# ---------------------------------------------------

def rag_pipeline(input_data):
    question = input_data["input"]
    chat_history = input_data["chat_history"]

    # ✅ Step 1: rewrite question
    new_question = rewrite_chain.invoke({
        "input": question,
        "chat_history": chat_history
    })

    # ✅ Step 2: retrieve docs
    docs = retriever.invoke(new_question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # ✅ Step 3: answer
    answer = qa_chain.invoke({
        "input": question,
        "context": context,
        "chat_history": chat_history
    })

    return answer

# ---------------------------------------------------
# ✅ Streamlit UI (with history)
# ---------------------------------------------------




if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask your question:")

if query:
    response = rag_pipeline({
        "input": query,
        "chat_history": st.session_state.chat_history
    })

    # ✅ Save history
    st.session_state.chat_history.append(HumanMessage(content=query))
    st.session_state.chat_history.append(AIMessage(content=response))

# ✅ Show history
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.write(f"🧑: {msg.content}")
    else:
        st.write(f"🤖: {msg.content}")
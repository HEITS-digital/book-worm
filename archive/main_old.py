# %%
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.llms import LlamaCpp
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
# %%
def book_questions_chain(model, retriever):
    claim_template = """You have the role of a well read librarian. Answer in 2-3 short sentences the question based only on the following context:
    {context}

    Question: {question}
    """
    book_question_prompt = ChatPromptTemplate.from_template(claim_template)

    claim_chain = (
        {"question": RunnablePassthrough(), "context": retriever}
        | book_question_prompt
        | model
        | StrOutputParser()
    )
    return claim_chain

# %%
MODEL_PATH = "models/firefly-llama2-7b-chat.Q5_K_M.gguf"
EMBEDDING_MODEL_PATH = "all-mpnet-base-v2"
SENTENCE_TRANSFORMERS_HOME="/Users/gianistatie/Documents/ecree/models/embeddings"

# Make sure the model path is correct for your system!
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH, cache_folder=SENTENCE_TRANSFORMERS_HOME)

model = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.75,
    max_tokens=1000,
    n_ctx=1024,
    top_p=1,
    verbose=True,  # Verbose is required to pass to the callback manager
)
# %%
data = open("books/pg55317.txt", "r").read()
data = [line.replace("\n", " ") for line in data.split("\n\n")]
data = [line for line in data if line != line.upper()]

vectorstore = FAISS.from_texts(
    data, embedding=embedding_model
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
# %%
query = "What about game development? Can the book help me become better at game development?"

chain = book_questions_chain(model, retriever)
answer = chain.invoke(query)

# %%
print(answer)
# %%

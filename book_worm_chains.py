from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

def fetch_summarizing_chain(llm):
    # Define prompt
    prompt_template = """Summarize in maximum 3 sentence paragraph the main ideas within the following text:
    "{text}"

    SUMMARY:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    
    return stuff_chain

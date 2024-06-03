import re
import json
from typing import Optional

import spacy
import numpy as np
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama, LogitsProcessorList
from lmformatenforcer import CharacterLevelParser, JsonSchemaParser
from lmformatenforcer.integrations.llamacpp import build_llamacpp_logits_processor, build_token_enforcer_tokenizer_data

from src.utils.lexrank import degree_centrality_scores
from src.utils.prompts import get_summarization_prompt

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

from src.utils.utils import llamacpp_with_character_level_parser

SENTENCE_TRANSFORMERS_HOME = "./models/embeddings"
MODEL_PATH = "models/llama-2-13b-chat.Q2_K.gguf"
EMBEDDING_MODEL_PATH = "all-mpnet-base-v2"


class BookWorm:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

        self.embedding_model = SentenceTransformer(
            model_name_or_path=EMBEDDING_MODEL_PATH,
            cache_folder=SENTENCE_TRANSFORMERS_HOME
        )

        self.model = Llama(
            model_path=MODEL_PATH,
            temperature=0.75,
            max_tokens=1000,
            n_ctx=1024,
            top_p=1,
            verbose=True,  # Verbose is required to pass to the callback manager
        )

        self.token_enforcer = build_token_enforcer_tokenizer_data(self.model)

    def get_central_sentences(self, text, k=10):
        # Split the document into sentences
        sentences = [x.text for x in self.nlp(text).sents]

        # Compute the sentence embeddings
        embeddings = self.embedding_model.encode(sentences)

        # Compute the pair-wise cosine similarities
        cos_scores = util.cos_sim(embeddings, embeddings).numpy()

        # Compute the centrality for each sentence
        centrality_scores = degree_centrality_scores(cos_scores, threshold=None)

        # We argsort so that the first element is the sentence with the highest score
        most_central_sentence_indices = np.argsort(-centrality_scores)

        # Get sentence that have the highest central index
        sentences = [sentences[idx].strip() for idx in most_central_sentence_indices[:k]]
        return sentences

    def summarize_sentences(self, sentences):
        text_to_summarize = " ".join(sentences[:4])  # TODO: find a better way to manage max_token limitation
        summarization_prompt = get_summarization_prompt(text_to_summarize)

        output = llamacpp_with_character_level_parser(
            summarization_prompt["prompt"], self.model, self.token_enforcer,
            JsonSchemaParser(summarization_prompt["output_schema"])
        )

        return json.loads(output)

    def _llamacpp_with_character_level_parser(self, prompt: str,
                                              character_level_parser: Optional[CharacterLevelParser]) -> str:
        logits_processors: Optional[LogitsProcessorList] = None
        if character_level_parser:
            logits_processors = LogitsProcessorList(
                [build_llamacpp_logits_processor(self.token_enforcer, character_level_parser)])

        output = self.model(prompt, temperature=0.0, logits_processor=logits_processors, max_tokens=1024)
        text: str = output['choices'][0]['text']
        return text

    @staticmethod
    def get_relevant_content(documents, embedding_model, question, chunk_size):
        text_splitter = CharacterTextSplitter(separator=".", chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(documents)
        db = Chroma.from_documents(documents, embedding_model)
        docs = db.similarity_search(question)
        page_contents = [doc.page_content for doc in docs]
        return page_contents

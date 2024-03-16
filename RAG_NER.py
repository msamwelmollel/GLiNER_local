# -*- coding: utf-8 -*-
"""
Created on Thu Mar  11 03:07:05 2024

@author: msamwelmollel
"""




#Important install 
#!pip install gliner

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))




from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.together import TogetherEmbedding



import json
from model import GLiNER





# model = GLiNER.from_pretrained("urchade/gliner_base", local_files_only=True)

model = GLiNER.from_pretrained("urchade/gliner_multi", local_files_only=True)


text = """
"In the recent advancements and initiatives related to water conservation and sustainability, how has Jordan's work influenced policies in the Jordan Valley, and what role does the Jordan brand play in these environmental efforts, considering the support from the Jordan River Foundation?"

"""

text2 = text


# Replace 'path_to_your_json_file.json' with the actual file path
with open('label_entity.json', 'r') as file:
    labels = json.load(file)



entities = model.predict_entities(text, labels)


# Function to refine the text by adding entity labels
def refine_text_with_entities(text, entities):
    sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
    for entity in sorted_entities:
        text = text[:entity['start']] + f"[{entity['text']}: {entity['label']}] " + text[entity['end']:]
    return text

# Refine the original text
refined_text = refine_text_with_entities(text, entities)
print(refined_text)

path_to_local_GGUF= "mistral-7b-instruct-v0.1.Q4_K_M.gguf"

llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    # model_url=None,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=path_to_local_GGUF,
    temperature=0.1 ,
    max_new_tokens=500,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=2000,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 20},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

TogetherAPI = 'xxx'

Settings.embed_model = TogetherEmbedding(
    model_name="togethercomputer/m2-bert-80M-8k-retrieval", api_key= TogetherAPI
)
Settings.llm = llm


documents = SimpleDirectoryReader("data").load_data()

index = VectorStoreIndex.from_documents(documents)


query_engine = index.as_query_engine()


answer_refined_query = query_engine.query(refined_text)


answer_original_query = query_engine.query(text2)






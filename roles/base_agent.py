import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
import torch
from uuid import uuid4
import json
from openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
import pandas as pd
from datasets import load_dataset

class BaseAgent(object):
    def __init__(self, agent_name:str, model_id:str, sys_prompt:str, quantization:bool=False, **kwargs) -> None:
        # init model and tokenizer
        if quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                trust_remote_code=True
            )
        self.model = self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # set work directory
        self.work_dir = kwargs.get("work_dir", f"./Case_{uuid4()}")
        self.cache_dir = os.path.join(self.work_dir, "Cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # set message cache file
        self.agent_name = agent_name
        self.msg_cache_file = os.path.join(self.cache_dir, self.agent_name + "_msg.jsonl")
        if os.path.exists(self.msg_cache_file):
            self.load_message()
        else:
            # set system prompt
            if sys_prompt:
                self.sys_prompt = {"role": "system", "content": sys_prompt}
                with open(self.msg_cache_file, 'a') as f:
                    f.write(f"{json.dumps(self.sys_prompt)}\n")
                self.messages = [self.sys_prompt]
            else:
                self.messages = []
        
        # set role cache file
        self.role_cache_file = os.path.join(self.cache_dir, self.agent_name + ".json")
        if os.path.exists(self.role_cache_file):
            with open(self.role_cache_file, 'r') as f:
                self.role = json.load(f)
        else:
            self.role = {"work_dir": self.work_dir}
    
    def load_knowledge(self, dataset_path:str) -> None:
        df = pd.read_json(dataset_path)
        df_loader = DataFrameLoader(df, page_content_column="requirement")
        df_doc = df_loader.load()
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_db = Chroma.from_documents(documents=df_doc, 
                                               embedding=embeddings, 
                                               persist_directory="chroma_db")
        self.retriever = self.vector_db.as_retriever()

    def retrieve(self, query:str):
        # retrieve top k similar documents to query
        docs = self.retriever.get_relevant_documents(query)
        return docs[:2]

    def combine_prompt(self, c_prompt:dict, combine:bool=True) -> None:
        self.messages.append(c_prompt)
        if combine:
            with open(self.msg_cache_file, 'a') as f:
                f.write(f"{json.dumps(c_prompt)}\n")

    def chat(self, max_token:int=512) -> str:
        # tokenize the messages
        inputs = self.tokenizer.apply_chat_template(self.messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        # generate response
        output = self.model.generate(
            inputs,
            max_new_tokens=max_token,
            do_sample=False,
            num_return_sequences=1,
            top_k=50,
            top_p=0.95,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        output = output[0][len(inputs[0]):].to("cpu")
        # decode the response
        response = self.tokenizer.decode(output)
        return response

    def load_message(self) -> None:
        self.messages = []
        with open(self.msg_cache_file, 'r') as f:
            for line in f:
                self.messages.append(json.loads(line))

    def dump_role_cache(self) -> None:
        with open(self.role_cache_file, 'w') as f:
            json.dump(self.role, f)
    
    def dump_code(self) -> None:
        if "code" in self.role and "filename" in self.role:
            with open(os.path.join(self.work_dir, self.role["filename"]), "w") as f:
                f.write(self.role["code"])

    def dump_files(self) -> None:
        self.dump_role_cache()
        self.dump_code()

    def release(self):
        self.model = None
        self.tokenizer = None
        gc.collect()
        # clear cuda cache
        torch.cuda.empty_cache()


class BaseGPTAgent(BaseAgent):
    def __init__(self, model_id:str, sys_prompt:str, **kwargs) -> None:
        with open("./api_key", "r") as f:
            api_key = f.read().strip()
        self.client = OpenAI(api_key=api_key)
        self.model_id = model_id
        
        # set work directory
        self.work_dir = kwargs.get("work_dir", f"./Case_{uuid4()}")
        self.cache_dir = os.path.join(self.work_dir, "Cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # set message cache file
        self.msg_cache_file = os.path.join(self.cache_dir, kwargs.get("agent_name", "DefaultAgent") + "_msg.jsonl")
        if os.path.exists(self.msg_cache_file):
            self.load_message()
        else:
            # set system prompt
            if sys_prompt:
                self.sys_prompt = {"role": "system", "content": sys_prompt}
                with open(self.msg_cache_file, 'a') as f:
                    f.write(f"{json.dumps(self.sys_prompt)}\n")
                self.messages = [self.sys_prompt]
            else:
                self.messages = []
        
        # set role cache file
        self.role_cache_file = os.path.join(self.cache_dir, kwargs.get("agent_name", "DefaultAgent") + ".json")
        if os.path.exists(self.role_cache_file):
            with open(self.role_cache_file, 'r') as f:
                self.role = json.load(f)
        else:
            self.role = {"work_dir": self.work_dir}

    def chat(self) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=self.messages
        )
        return response.choices[0].message.content.strip()

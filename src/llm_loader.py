from langchain_community.chat_models import ChatOllama


class LLM : 
    def __init__(self) : 
        self.params = {   
            "max_new_tokens": 2048,
            "temperature": 0.3,
            "top_p": 0.4,
            "top_k": 40,
            "typical_p": 0.95,
            "min_p": 0,
            "repetition_penalty": 1.17,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "mirostat_mode": 2,
            "mirostat_tau": 4,
            "mirostat_eta": 0.1,
            "num_ctx": 8192,
            "seed": -1,
            "tfs": 1,
        }
    
    def load_llm(self):
        llm = ChatOllama(model="EEVE-K", **self.params)
        return llm
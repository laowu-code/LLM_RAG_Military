from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
device = "cuda" # the device to load the model onto
import torch
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from vllm import LLMEngine, SamplingParams
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from vllm.outputs import RequestOutput
import time
import os
import re
import torch
import argparse
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from typing import List, Tuple
import numpy as np
from langchain.document_loaders import TextLoader,PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
import streamlit as st
from transformers import TextStreamer
from transformers import TextIteratorStreamer
from threading import Thread
import asyncio
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
tokenizer = AutoTokenizer.from_pretrained('qwen/autodl-tmp/qwen/Qwen2-7B-Instruct/', use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('qwen/autodl-tmp/qwen/Qwen2-7B-Instruct/', device_map="auto",torch_dtype="auto")
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

class TextChunk:
    def __init__(self, text):
        self.text = text

class Qwen(LLM, ABC):
     max_token: int = 10000
     temperature: float = 0.01
     top_p = 0.9
     history_len: int = 100
     conversation_history: List[str] = Field(default_factory=list)
     memory_size: int = 1024
     def __init__(self, memory_size=1024):
        super().__init__()
        # self.conversation_history = []
        # self.memory_size = memory_size 

     @property
     def _llm_type(self) -> str:
         return "Qwen"

     @property
     def _history_len(self) -> int:
         return self.history_len

     def set_history_len(self, history_len: int = 10) -> None:
         self.history_len = history_len

     def _call(
         self,
         prompt: str,
         stop: Optional[List[str]] = None,
         run_manager: Optional[CallbackManagerForLLMRun] = None,
     ) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            streamer=streamer,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

     @property
     def _identifying_params(self) -> Mapping[str, Any]:
         """Get the identifying parameters."""
         return {"max_token": self.max_token,
                 "temperature": self.temperature,
                 "top_p": self.top_p,
                 "history_len": self.history_len}
     def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ):
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Prepare the text input
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        # Define generation parameters
        generation_kwargs = {
            "input_ids": model_inputs.input_ids,
            "max_new_tokens": 4096,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "streamer": streamer
        }

        # Run generation in a background thread to avoid blocking
        # def generate_in_background():
        #     model.generate(**generation_kwargs)
        print("stream:\n")
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        # Collect and yield generated text
        response_text = ""
        count=0
        for new_text in streamer:
            print(count)
            response_text += new_text
            yield TextChunk(new_text)
            count+=1

class QwenWithMemory(Qwen):
    conversation_history: List[str] = Field(default_factory=list)
    memory_size: int = 4096
    def __init__(self, memory_size=4096):
        super().__init__()
        self.conversation_history = []
        self.memory_size = memory_size  # Set memory size in tokens or characters

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        self.conversation_history.append({"role": "user", "content": prompt})
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.conversation_history])
        history_text = truncate_history_by_chars(history_text, max_chars=self.memory_size)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": history_text}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            streamer=streamer,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Add the model's response to the conversation history
        self.conversation_history.append({"role": "assistant", "content": response})

        return response


    

class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", "\n", text)
            text = re.sub('\s', ' ', text)
            text = text.replace("\n\n", "")
        sent_sep_pattern = re.compile(
            '([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)
        return sent_list


# def load_file(filepath):
#     loader = TextLoader(filepath, autodetect_encoding=True)
#     textsplitter = ChineseTextSplitter(pdf=False)
#     docs = loader.load_and_split(textsplitter)
#     write_check_file(filepath, docs)
#     return docs

def load_file(filepath,size_chunk=256,overlap=50):
    docs = []
    if os.path.isdir(filepath):
        for file in os.listdir(filepath):
            file_path = os.path.join(filepath, file)
            if file.lower().endswith('.pdf'):
                loader = PyMuPDFLoader(file_path)
                textsplitter = ChineseTextSplitter(pdf=True,chunk_size=size_chunk,chunk_overlap=overlap)
            elif file.lower().endswith('.txt'):
                loader = TextLoader(file_path, autodetect_encoding=True)
                textsplitter = ChineseTextSplitter(pdf=False,chunk_size=size_chunk,chunk_overlap=overlap)
            else:
                continue
            docs.extend(loader.load_and_split(textsplitter))
    else:
        if filepath.lower().endswith('.pdf'):
            loader = PyMuPDFLoader(filepath)
            textsplitter = ChineseTextSplitter(pdf=True,chunk_size=size_chunk,chunk_overlap=overlap)
        elif filepath.lower().endswith('.txt'):
            loader = TextLoader(filepath, autodetect_encoding=True)
            textsplitter = ChineseTextSplitter(pdf=False,chunk_size=size_chunk,chunk_overlap=overlap)
        else:
            raise ValueError("Unsupported file format")
        docs.extend(loader.load_and_split(textsplitter))
    write_check_file(filepath, docs)
    torch.cuda.empty_cache()
    return docs



def write_check_file(filepath, docs):
    folder_path = os.path.join(os.path.dirname(filepath), "tmp_files")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fp = os.path.join(folder_path, 'load_file.txt')
    with open(fp, 'a+', encoding='utf-8') as fout:
        fout.write("filepath=%s,len=%s" % (filepath, len(docs)))
        fout.write('\n')
        for i in docs:
            fout.write(str(i))
            fout.write('\n')
        fout.close()


def separate_list(ls: List[int]) -> List[List[int]]:
    lists = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i - 1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists


class FAISSWrapper(FAISS):
    chunk_size = 500
    chunk_conent = True
    score_threshold = 0

    def similarity_search_with_score_by_vector(
            self, embedding: List[float], k: int = 4
    ) -> List[Tuple[Document, float]]:
        scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
        docs = []
        id_set = set()
        store_len = len(self.index_to_docstore_id)
        for j, i in enumerate(indices[0]):
            if i == -1 or 0 < self.score_threshold < scores[0][j]:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            if not self.chunk_conent:
                if not isinstance(doc, Document):
                    raise ValueError(f"Could not find document for id {_id}, got {doc}")
                doc.metadata["score"] = int(scores[0][j])
                docs.append(doc)
                continue
            id_set.add(i)
            docs_len = len(doc.page_content)
            for k in range(1, max(i, store_len - i)):
                break_flag = False
                for l in [i + k, i - k]:
                    if 0 <= l < len(self.index_to_docstore_id):
                        _id0 = self.index_to_docstore_id[l]
                        doc0 = self.docstore.search(_id0)
                        if docs_len + len(doc0.page_content) > self.chunk_size:
                            break_flag = True
                            break
                        elif doc0.metadata["source"] == doc.metadata["source"]:
                            docs_len += len(doc0.page_content)
                            id_set.add(l)
                if break_flag:
                    break
        if not self.chunk_conent:
            return docs
        if len(id_set) == 0 and self.score_threshold > 0:
            return []
        id_list = sorted(list(id_set))
        id_lists = separate_list(id_list)
        for id_seq in id_lists:
            for id in id_seq:
                if id == id_seq[0]:
                    _id = self.index_to_docstore_id[id]
                    doc = self.docstore.search(_id)
                else:
                    _id0 = self.index_to_docstore_id[id]
                    doc0 = self.docstore.search(_id0)
                    doc.page_content += " " + doc0.page_content
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            doc_score = min([scores[0][id] for id in [indices[0].tolist().index(i) for i in id_seq if i in indices[0]]])
            doc.metadata["score"] = int(doc_score)
            docs.append((doc, doc_score))
        return docs

def create_faiss_index(docs, embeddings,index_path='faiss_index', batch_size=10):
    index = None
    for i in tqdm(range(0, len(docs), batch_size), desc="Building FAISS Index"):
        batch_docs = docs[i:i + batch_size]
        batch_embeddings = embeddings.embed_documents([doc.page_content for doc in batch_docs])
        if index is None:
            index = FAISS.from_documents(batch_docs, embeddings)
        else:
            index.add_documents(batch_docs)
        torch.cuda.empty_cache()  # 清理GPU内存缓存
    # faiss.write_index(index.index, index_path+'/faiss.index')
    
    # # Save the documents to disk
    # with open(index_path + "/faiss_docs.pkl", "wb") as f:
    #     pickle.dump(docs, f)
    return index

# def load_faiss_index(index_path):
#     # Load the FAISS index from disk
#     index = faiss.read_index(index_path+'/faiss.index')
    
#     # Load the documents from disk
#     with open(index_path + "/faiss_docs.pkl", "rb") as f:
#         docs = pickle.load(f)
    
#     # Create a FAISSWrapper instance with the loaded index and documents
#     faiss_wrapper = FAISS(index=index, docstore={i: doc for i, doc in enumerate(docs)}, embeddings=None)
    
#     return faiss_wrapper
def truncate_history(history, max_tokens=1024):
    # 按 token 数量截断历史记录，这里假设每个字符约为 0.5 个 token
    tokens = history.split()
    if len(tokens) > max_tokens:
        # 只保留最后 max_tokens 个 token
        truncated_history = ' '.join(tokens[-max_tokens:])
    else:
        truncated_history = history
    return truncated_history

# 或者按字符长度截断
def truncate_history_by_chars(history, max_chars=4096):
    # 如果历史记录超过了指定的字符数，截断最早的部分
    if len(history) > max_chars:
        truncated_history = history[-max_chars:]
    else:
        truncated_history = history
    return truncated_history



class RAGResponse(BaseModel):
    question: str = Field(description="Questions from users")
    answer: str = Field(description="Answers based on known information")
    # status: str = Field(description="回答的状态")

# output_schema = text.schema("response", {
#     "question": str,
#     "answer": str,
#     "context_used": list,
#     "status": str
# })
class ContextUsed(BaseModel):
    source: str = Field(description="文档片段的来源或标识")
    content: str = Field(description="与回答相关的文档内容")

class JsonResponse(BaseModel):
    question: str = Field(description="用户的提问")
    answer: str = Field(description="基于已知信息的回答")
    context_used: List[ContextUsed] = Field(description="与回答相关的文档片段列表")
    status: str = Field(description="回答的状态")


def stream_response(prompt, rag_chain):
    response_text = ""  # 初始化response_text
    response_container = st.empty()  # 创建一个空容器
    # 假设 rag_chain.stream 返回一个生成器，可以逐步生成文本
    for chunk in rag_chain.invoke(prompt):
        response_text += chunk
        response_container.write(response_text)  # 动态更新网页内容
    return response_text

if __name__ == '__main__':
    # load docs (pdf file or txt file)
    # qa = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type=CHAIN_TYPE,
    #     retriever=retriever_,
    #     chain_type_kwargs=chain_type_kwargs,
    #     )
    # qa = RetrievalQA.from_chain_type(
    # llm=llm,
    # chain_type=CHAIN_TYPE,
    # retriever=retriever_,
    # chain_type_kwargs={"prompt": PROMPT_TEMPLATE},
    # return_source_documents=True  # 返回与答案相关的文档片段
# )
    # parser_tmp = JsonOutputParser()
        # Streamlit应用代码
    # 在侧边栏中创建一个标题和一个链接
    with st.sidebar:
        st.markdown("# RAG Chatbot")
        st.markdown("# Qwen2-7B-Insturct")
        # max_length = st.slider("max_length", 0, 1024, 512, step=1)

    # 创建一个标题和一个副标题
    st.title("💬 RAG Chatbot")
    st.caption("🚀 A streamlit chatbot")



    @st.cache_resource
    def get_rag():
        # 初始化你的RAG系统，这里只是一个示例
        # 初始化模型、向量检索器等
        conversation_history = ""
        max_history_chars = 4096  # 设置最大字符长度限制
        filepath = 'data_pdf'
        # Embedding model name
        EMBEDDING_MODEL = 'text2vec'
        # Conversation history:
        # {history}
        # {format_instructions}
        PROMPT_TEMPLATE = """Known information:
        {context}
        Based on the above known information, respond to the user's question concisely \
        and professionally. If an answer cannot be derived from it,\
        say 'The question cannot be answered with the given information' or \
        'Not enough relevant information has been provided,' and do not \
        include fabricated details in the answer. Please respond in Chinese. \
        The question is {question}"""

        system_prompt = (
            "Based on the above known information, respond to the user's question concisely \
        and professionally. If an answer cannot be derived from it,\
        say 'The question cannot be answered with the given information' or \
        'Not enough relevant information has been provided,' and do not \
        include fabricated details in the answer. Please respond in Chinese."
            "\n\n"
            "{context}")

        # qa_prompt = ChatPromptTemplate.from_messages(
        #     [
        #         ("system", system_prompt),
        #         MessagesPlaceholder("chat_history"),
        #         ("human", "{question}"),
        #     ]
        # )
        # Embedding running device
        EMBEDDING_DEVICE = "cuda"
        # return top-k text chunk from vector store
        VECTOR_SEARCH_TOP_K = 5
        CHAIN_TYPE = 'stuff'
        llm = Qwen()
        # llm=QwenWithMemory(memory_size=1024)
        embeddings = HuggingFaceEmbeddings(model_name="embeddings/text2vec-large-chinese",model_kwargs={'device': 'cpu'})
        INDEX_PATH='faiss_index'
        # docs = load_file(filepath)
        # faiss_index = create_faiss_index(docs, embeddings)
        # faiss_index.save_local("faiss_index")
        faiss_index = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
        # parser_rag = JsonOutputParser(pydantic_object=RAGResponse)
        # output_parser = PydanticOutputParser(pydantic_object=JsonResponse)
        prompt = PromptTemplate(template=PROMPT_TEMPLATE, 
        input_variables=["context", "question"],
        # partial_variables={"format_instructions": parser_rag.get_format_instructions()},
        )
        # chain_type_kwargs = {"prompt": prompt, "document_variable_name": "context"}
        retriever_ = faiss_index.as_retriever(search_kwargs={"k": VECTOR_SEARCH_TOP_K})
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        rag_chain = ({"context": retriever_ | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    )  # 你已经设置好的RAG链
        return rag_chain

    # 初始化RAG系统
    rag_chain = get_rag()
    # msgs = StreamlitChatMessageHistory(key="langchain_messages")
    # if len(msgs.messages) == 0:
    #     msgs.add_ai_message("有什么可以帮您的？")
#     chain_with_history = RunnableWithMessageHistory(
#     rag_chain,
#     lambda session_id: msgs,
#     input_messages_key="question",
#     history_messages_key="chat_history",
# )
    print('Ready')
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "有什么可以帮您的？"}]
    # 遍历session_state中的所有消息，并显示在聊天界面上
    for msg in st.session_state['messages']:
        st.chat_message(msg["role"]).write(msg["content"])
    # 如果用户在聊天输入框中输入了内容，则执行以下操作
    if prompt := st.chat_input():
        # 将用户的输入添加到session_state中的messages列表中
        st.session_state.messages.append({"role": "user", "content": prompt})
        # 在聊天界面上显示用户的输入
        st.chat_message("user").write(prompt)
        # response=rag_chain.invoke(prompt)
        st.session_state.messages.append({"role": "assistant", "content": ""})
        # 调用stream_response，实现流式输出
        response=stream_response(prompt,rag_chain)
        # st.chat_message("assistant").write(response)
        # 在最后更新st.session_state
        


    # while True:
    #     query = input('用户输入：')
    #     start_=time.time()
    #     anserws = rag_chain.invoke(query)
    #     # for s in rag_chain.stream(query):
    #     #     print(s)
    #     end_=time.time()
    #     print(f'运行时间：{(end_-start_)}s\n')
    #     json_output = output_parser.parse({
    #     "question": query,
    #     "answer": response['result'],
    #     "context_used": [
    #         {
    #             "source": doc.metadata.get("source", "unknown"),
    #             "content": doc.page_content
    #         } for doc in response['source_documents']
    #     ],
    #     "status": "success" if response['result'] else "unable to answer"
    # })
    
    # # 输出 JSON
    # print(json_output.json(indent=4, ensure_ascii=False))
        # conversation_history += f"User: {query}\nAI: {answer}\n"
        # conversation_history = truncate_history_by_chars(conversation_history, max_chars=max_history_chars)

# streamlit run Qwen2_RAG.py --server.address 127.0.0.1 --server.port 6006


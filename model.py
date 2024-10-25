from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import WikipediaRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
import torch

class LanguageModelPipeline:
    def __init__(self, model_embedding_name):
        self.model_embedding_name = model_embedding_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embeddings = self.create_embedding_model()

    def get_embedding(self):
        return self.embeddings

    def create_embedding_model(self):
        model_kwargs = {'device': self.device}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=self.model_embedding_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        # embedder = CacheBackedEmbeddings.from_bytes_store(embeddings, self.store, namespace=self.model_embedding_name)
        return embeddings

    def create_prompt(self, template):
        prompt = PromptTemplate(
            template=template,
            input_variables=['context', 'question'],
        )
        return prompt

    def create_chain_reranking(self, llm, prompt, retriever_db, return_source_documents=True):
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=retriever_db,
            return_source_documents=return_source_documents,
            chain_type_kwargs={
                'prompt': prompt,
                'document_variable_name': 'context',  
            },
        )
        return chain
    
    def create_chain_wiki(self, llm, prompt, top_k_documents=3, return_source_documents=True):
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=WikipediaRetriever(
                top_k_results=top_k_documents,
                lang='vi',
                doc_content_chars_max=2048
                ),
            return_source_documents=return_source_documents,
            chain_type_kwargs={
                'prompt': prompt,
                'document_variable_name': 'context',  
            },
        )
        return chain
    
    def create_chain_hybird(self, llm, prompt, collection, db, top_k_documents=3, return_source_documents=True):
        docs = collection.find()
        documents = [
            Document(
                page_content=doc['text'], 
                metadata={
                    'page': doc['page'] if (doc.get('page')) else 1,
                    'source': doc['source'],
                    'source_type': doc['source_type'] if (doc.get('source_type')) else 'unknown',
                }
            ) for doc in docs
        ]

        bm25_retriever = BM25Retriever.from_documents(documents)
        semantic_retriever = db.as_retriever(search_kwargs={"k": top_k_documents})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, semantic_retriever], 
            weights=[0.4, 0.6]
        )

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=ensemble_retriever,
            return_source_documents=return_source_documents,
            chain_type_kwargs={
                'prompt': prompt,
                'document_variable_name': 'context',  
            },
        )

        return chain
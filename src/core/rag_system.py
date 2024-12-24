import torch
import logging
from typing import List, Dict
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from src.utils.constants import MODEL_NAME

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.embeddings = None
        self.vector_store = None
        self.qa_chain = None
        self.tokenizer = None
        self.model = None
        self.is_initialized = False
        self.processed_files = set()

    def initialize_model(self):
        try:
            # Configurar embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-large",
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Token de HuggingFace
            hf_token = os.environ.get('HUGGINGFACE_TOKEN')
            if not hf_token:
                raise ValueError("No Hugging Face token found")

            # Inicializar modelo y tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=hf_token,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=hf_token,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto"
            )
            
            # Pipeline de generación
            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.95,
                repetition_penalty=1.15,
                device_map="auto"
            )
            
            self.llm = HuggingFacePipeline(pipeline=pipe)
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Model initialization error: {str(e)}")
            raise

    def process_documents(self, files: List):
        """Procesa documentos y actualiza vector store"""
        [Implementación del procesamiento de documentos]

    def generate_response(self, question: str) -> Dict:
        """Genera respuesta basada en documentos procesados"""
        [Implementación de la generación de respuesta]
def process_documents(self, files: List):
    try:
        documents = []
        new_files = []
        
        # Procesar archivos nuevos
        for file in files:
            if file.name not in self.processed_files:
                docs = DocumentLoader.load_file(file.name)
                documents.extend(docs)
                new_files.append(file.name)
                self.processed_files.add(file.name)
        
        if not documents:
            return
            
        # Dividir documentos
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        # Crear/actualizar vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)
        
        # Configurar QA chain
        prompt_template = """
        Context: {context}
        Question: {question}
        Answer the question clearly based on the context.
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 6}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
    except Exception as e:
        logger.error(f"Document processing error: {str(e)}")
        raise

def generate_response(self, question: str) -> Dict:
    if not self.is_initialized or self.qa_chain is None:
        return {
            'answer': "Please upload documents first.",
            'sources': []
        }
    
    try:
        result = self.qa_chain({"query": question})
        
        response = {
            'answer': result['result'],
            'sources': [{
                'title': doc.metadata.get('title', 'Unknown'),
                'content': doc.page_content[:200] + "...",
                'metadata': doc.metadata
            } for doc in result['source_documents']]
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Response generation error: {str(e)}")
        raise
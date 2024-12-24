import os
import logging
from typing import List
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader, UnstructuredFileLoader

logger = logging.getLogger(__name__)

class DocumentLoader:
    @staticmethod
    def load_file(file_path: str) -> List:
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif ext in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            elif ext == '.csv':
                loader = CSVLoader(file_path)
            else:
                loader = UnstructuredFileLoader(file_path)
            
            documents = loader.load()
            for doc in documents:
                doc.metadata.update({
                    'title': os.path.basename(file_path),
                    'type': 'document',
                    'format': ext[1:]
                })
            return documents
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise
import os
from src.core.rag_system import RAGSystem
from src.interface.gradio_app import create_gradio_interface
from src.utils.logger import setup_logger
import spaces

logger = setup_logger()

@spaces.GPU(duration=60)
def main():
    try:
        logger.info("Initializing RAG system...")
        rag_system = RAGSystem()
        
        logger.info("Creating Gradio interface...")
        demo = create_gradio_interface(rag_system)
        demo.launch()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
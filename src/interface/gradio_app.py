import gradio as gr
from typing import List
import logging
from src.core.rag_system import RAGSystem
from src.utils.constants import SUPPORTED_FORMATS

logger = logging.getLogger(__name__)

def create_gradio_interface(rag_system: RAGSystem):
    with gr.Blocks(css="div.gradio-container {background-color: #f0f2f6}") as demo:
        gr.HTML("""
            <div style="text-align: center; max-width: 800px; margin: 0 auto; padding: 20px;">
                <h1 style="color: #2d333a;">📚 CV Assistant</h1>
                <p style="color: #4a5568;">AI Assistant for CV Analysis</p>
            </div>
        """)

        with gr.Row():
            files = gr.Files(
                label="Upload CV Documents",
                file_types=SUPPORTED_FORMATS,
                file_count="multiple"
            )

        chatbot = gr.Chatbot(
            show_label=False,
            height=500,
            bubble_full_width=True,
            show_copy_button=True
        )
        
        with gr.Row():
            message = gr.Textbox(
                placeholder="Ask about the CV...",
                show_label=False,
                container=False,
                scale=8
            )
            clear = gr.Button("🗑️ Clear", size="sm", scale=1)

        def process_response(user_input: str, chat_history: List, files: List) -> tuple:
            try:
                if not rag_system.is_initialized:
                    rag_system.initialize_model()
                
                if files:
                    rag_system.process_documents(files)
                    
                response = rag_system.generate_response(user_input)
                answer = response['answer']
                
                sources = set([source['title'] for source in response['sources'][:3]])
                if sources:
                    answer += "\n\n📚 Sources:\n" + "\n".join([f"• {source}" for source in sources])
                
                chat_history.append((user_input, answer))
                return chat_history
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                chat_history.append((user_input, f"Error: {str(e)}"))
                return chat_history

        def clear_context():
            rag_system.vector_store = None
            rag_system.processed_files.clear()
            return None

        message.submit(process_response, [message, chatbot, files], [chatbot])
        clear.click(clear_context, None, chatbot)

        return demo
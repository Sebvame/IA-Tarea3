# app.py
import streamlit as st
import os
import tempfile
from typing import List
import time

# Importar componentes
from components.rag_system import RAGSystem
from components.agent_controller import AgentController, check_ollama_connection
from utils.config import load_config

# Configuración de página
st.set_page_config(
    page_title="🤖 Asistente RAG Académico",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
.stChatMessage {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}

.user-message {
    background-color: #e3f2fd;
    border-left: 4px solid #2196f3;
}

.assistant-message {
    background-color: #f3e5f5;
    border-left: 4px solid #9c27b0;
}

.sidebar .stSelectbox {
    margin-bottom: 1rem;
}

.metric-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #dee2e6;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_system():
    """Inicializa el sistema RAG (cached)"""
    return RAGSystem()

@st.cache_resource
def initialize_agent(_rag_system):
    """Inicializa el agente (cached)"""
    return AgentController(_rag_system)

def save_uploaded_file(uploaded_file) -> str:
    """Guarda archivo subido temporalmente"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def main():
    # Título principal
    st.title("🤖 Asistente Conversacional RAG")
    st.markdown("*Asistente académico con búsqueda inteligente en documentos IEEE*")
    
    # Verificar Ollama
    if not check_ollama_connection():
        st.error("❌ Ollama no está disponible. Consulta las instrucciones de instalación.")
        st.stop()
    
    # Inicializar componentes
    rag_system = initialize_rag_system()
    
    # Sidebar para configuración
    with st.sidebar:
        st.header("📁 Gestión de Documentos")
        
        # Upload de archivos
        uploaded_files = st.file_uploader(
            "Sube documentos PDF",
            type=['pdf'],
            accept_multiple_files=True,
            help="Sube uno o más archivos PDF para crear la base de conocimientos"
        )
        
        # Procesamiento de documentos
        if uploaded_files:
            if st.button("🔄 Procesar Documentos"):
                with st.spinner("Procesando documentos..."):
                    # Guardar archivos temporalmente
                    temp_paths = []
                    for uploaded_file in uploaded_files:
                        temp_path = save_uploaded_file(uploaded_file)
                        temp_paths.append(temp_path)
                    
                    try:
                        # Construir índice
                        result = rag_system.build_index(temp_paths)
                        
                        st.success(f"✅ Procesados {result['total_documents']} documentos")
                        st.success(f"📊 Creados {result['total_chunks']} fragmentos")
                        
                        # Limpiar archivos temporales
                        for temp_path in temp_paths:
                            os.unlink(temp_path)
                        
                        # Reinicializar agente con nuevos datos
                        st.cache_resource.clear()
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error procesando documentos: {e}")
                        # Limpiar archivos temporales en caso de error
                        for temp_path in temp_paths:
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)
        
        # Cargar índice existente
        st.subheader("📚 Base de Conocimientos")
        
        if st.button("📂 Cargar Índice Existente"):
            if rag_system.load_index():
                st.success("✅ Índice cargado correctamente")
                st.cache_resource.clear()
                st.rerun()
            else:
                st.error("❌ No se pudo cargar el índice")
        
        # Estadísticas del sistema
        if rag_system.chunks:
            st.subheader("📊 Estadísticas")
            stats = rag_system.get_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documentos", stats['unique_documents'])
                st.metric("Fragmentos", stats['total_chunks'])
            
            with col2:
                st.metric("Autores", stats['unique_authors'])
                st.metric("Dimensiones", stats['index_dimension'])
            
            # Distribución por secciones
            if stats['sections_distribution']:
                st.subheader("📑 Distribución por Secciones")
                for section, count in stats['sections_distribution'].items():
                    st.write(f"**{section}**: {count} fragmentos")
        
        # Configuración del agente
        st.subheader("⚙️ Configuración")
        
        temperature = st.slider(
            "Temperatura del modelo",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="Controla la creatividad de las respuestas"
        )
        
        max_results = st.slider(
            "Máximo resultados RAG",
            min_value=1,
            max_value=10,
            value=3,
            help="Número de fragmentos a recuperar"
        )
        
        # Botón para limpiar chat
        if st.button("🗑️ Limpiar Conversación"):
            st.session_state.messages = []
            if 'agent' in st.session_state:
                st.session_state.agent.clear_memory()
            st.rerun()
    
    # Área principal de chat
    
    # Verificar si hay documentos cargados
    if not rag_system.chunks:
        st.info("👋 ¡Bienvenido! Sube algunos documentos PDF en la barra lateral para comenzar.")
        st.markdown("""
        ### 🚀 Cómo usar este asistente:
        
        1. **Sube documentos**: Usa la barra lateral para subir archivos PDF
        2. **Procesa documentos**: Haz clic en "Procesar Documentos" 
        3. **Haz preguntas**: Escribe tus consultas en el chat
        4. **Obtén respuestas**: El asistente buscará en tus documentos
        
        ### 💡 Tipos de preguntas que puedes hacer:
        - "¿Qué metodologías se proponen en los documentos?"
        - "¿Cuáles son los resultados principales?"
        - "¿Qué autores han trabajado en machine learning?"
        - "Explícame el algoritmo mencionado en el paper de Smith"
        """)
        return
    
    # Inicializar agente si hay documentos
    if 'agent' not in st.session_state:
        with st.spinner("Inicializando agente..."):
            st.session_state.agent = initialize_agent(rag_system)
    
    # Inicializar mensajes de chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "¡Hola! Soy tu asistente académico. Puedo ayudarte a buscar información en tus documentos IEEE. ¿En qué puedo ayudarte?"
            }
        ]
    
    # Mostrar historial de mensajes
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input de chat
    if prompt := st.chat_input("Escribe tu pregunta aquí..."):
        # Añadir mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Mostrar mensaje del usuario
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generar respuesta
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                response = st.session_state.agent.chat(prompt)
            
            st.markdown(response)
        
        # Añadir respuesta del asistente
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Footer con información
    st.markdown("---")
    st.markdown(
        "🔧 **Tecnologías**: Streamlit + LangChain + Ollama + FAISS + Sentence Transformers | "
        "💡 **Modelos gratuitos**: Llama 3.2 + Multilingual MiniLM"
    )

if __name__ == "__main__":
    main()
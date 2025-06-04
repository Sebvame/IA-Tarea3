# components/agent_controller.py
from typing import List, Dict, Any, Optional, Type
from langchain_ollama import ChatOllama
from langchain.tools import BaseTool
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
import requests
from pydantic import Field

class RAGTool(BaseTool):
    """Herramienta para búsqueda RAG"""
    
    # ✅ Campos con anotaciones de tipo correctas
    name: str = Field(default="search_documents")
    description: str = Field(default="""
    Busca información en la base de conocimientos de documentos académicos.
    Usa esta herramienta cuando el usuario pregunte sobre conceptos, teorías, 
    metodologías que puedan estar en los documentos almacenados.
    Input: Una consulta de texto sobre el contenido académico.
    """)
    rag_system: Any = Field(description="Sistema RAG para búsqueda de documentos")
    
    def __init__(self, rag_system, **kwargs):
        super().__init__(rag_system=rag_system, **kwargs)
    
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Ejecuta búsqueda RAG"""
        try:
            results = self.rag_system.hybrid_search(query, k=3)
            
            if not results:
                return "No encontré información relevante en los documentos para esta consulta."
            
            # Formatear resultados
            context = "📚 Información encontrada en los documentos:\n\n"
            for i, result in enumerate(results, 1):
                context += f"**{i}. [{result['metadata'].get('section', 'Sin sección')}]** "
                context += f"(Relevancia: {result['similarity_score']:.2f})\n"
                context += f"{result['content'][:400]}...\n\n"
                
                # Agregar información del documento fuente
                if 'file_name' in result['metadata']:
                    context += f"📄 *Fuente: {result['metadata']['file_name']}*\n"
                if 'authors' in result['metadata']:
                    context += f"✍️ *Autores: {result['metadata']['authors']}*\n"
                context += "---\n\n"
            
            return context
            
        except Exception as e:
            return f"❌ Error en búsqueda de documentos: {str(e)}"
    
    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Versión asíncrona de la búsqueda RAG"""
        # Para simplicidad, llamamos la versión síncrona
        return self._run(query, run_manager=run_manager)

class WebSearchTool(BaseTool):
    """Herramienta para búsqueda web"""
    
    # ✅ Campos con anotaciones de tipo correctas
    name: str = Field(default="search_web")
    description: str = Field(default="""
    Busca información actualizada en Internet.
    Usa esta herramienta SOLO cuando el usuario solicite explícitamente 
    información actual, noticias recientes, o datos que no están en 
    los documentos almacenados.
    Input: Una consulta sobre información actual o reciente.
    """)
    
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Simula búsqueda web (reemplazar con Tavily API real si es necesario)"""
        # Para este ejemplo, simulamos una búsqueda web
        # En implementación real, integrar con Tavily API
        return f"""🌐 Búsqueda web realizada para: "{query}"

📋 **Nota importante**: Esta es una búsqueda simulada para demostración. 

🔧 **Para habilitar búsqueda web real**:
1. Registrarse en Tavily API (tavily.com)
2. Obtener API key gratuita
3. Configurar en archivo .env: TAVILY_API_KEY=tu_key
4. Descomentar código de integración en esta herramienta

💡 **Sugerencia**: Reformula tu pregunta para buscar en los documentos académicos cargados, 
donde tengo acceso completo al contenido."""
    
    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Versión asíncrona de la búsqueda web"""
        return self._run(query, run_manager=run_manager)

class DirectAnswerTool(BaseTool):
    """Herramienta para respuestas directas"""
    
    # ✅ Campos con anotaciones de tipo correctas
    name: str = Field(default="direct_answer")
    description: str = Field(default="""
    Responde directamente preguntas generales, saludos, o consultas que no requieren 
    búsqueda en documentos específicos.
    Usa esta herramienta para: saludos, preguntas sobre el sistema, 
    consultas generales de conocimiento común.
    Input: Pregunta general o saludo.
    """)
    
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Proporciona respuestas directas para consultas generales"""
        query_lower = query.lower()
        
        # Detectar tipos de consultas
        if any(greeting in query_lower for greeting in ['hola', 'hello', 'hi', 'buenas', 'saludos']):
            return """¡Hola! 👋 Soy tu asistente académico especializado en análisis de documentos IEEE. 

🎯 **Puedo ayudarte con**:
- Buscar información específica en tus documentos cargados
- Analizar metodologías y resultados de investigación
- Encontrar referencias y citas relevantes
- Explicar conceptos técnicos presentes en los papers

📝 **Ejemplos de preguntas**:
- "¿Qué metodologías proponen los autores?"
- "¿Cuáles son los principales resultados experimentales?"
- "¿Quién escribió sobre machine learning?"

¡Hazme cualquier pregunta sobre tus documentos académicos!"""
        
        elif any(word in query_lower for word in ['gracias', 'thanks', 'thank you']):
            return "¡De nada! 😊 Estoy aquí para ayudarte con cualquier consulta sobre tus documentos académicos."
        
        elif any(word in query_lower for word in ['ayuda', 'help', 'cómo', 'how']):
            return """🤖 **Guía de uso del asistente**:

1️⃣ **Sube documentos**: Usa la barra lateral para cargar PDFs
2️⃣ **Haz preguntas específicas**: Pregunta sobre contenido, autores, metodologías
3️⃣ **Explora resultados**: Revisa las fuentes que cito en mis respuestas

💡 **Consejos para mejores resultados**:
- Sé específico en tus preguntas
- Menciona términos técnicos relevantes
- Pregunta sobre aspectos concretos de los papers

¿Hay algo específico sobre lo que te gustaría preguntar?"""
        
        else:
            return f"""Entiendo que preguntas sobre: "{query}"

Para darte la mejor respuesta posible, necesito buscar en tus documentos académicos. 
¿Podrías reformular tu pregunta de manera más específica sobre el contenido de los papers?

💡 **Ejemplo**: En lugar de preguntas muy generales, puedes preguntar:
- "¿Qué algoritmos de machine learning mencionan los documentos?"
- "¿Cuáles son las limitaciones identificadas en los estudios?"
- "¿Qué trabajos futuros proponen los autores?"
"""
    
    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Versión asíncrona de respuesta directa"""
        return self._run(query, run_manager=run_manager)

class AgentController:
    """Controlador principal del agente conversacional"""
    
    def __init__(self, rag_system, model_name: str = "llama3.2:3b"):
        self.rag_system = rag_system
        
        # Inicializar LLM (Ollama)
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1,
            num_predict=1024,
            # Configuraciones adicionales para mejor rendimiento
            top_k=10,
            top_p=0.9,
        )
        
        # Crear herramientas con la sintaxis correcta
        self.tools = [
            RAGTool(rag_system=rag_system),
            WebSearchTool(),
            DirectAnswerTool()
        ]
        
        # Configurar memoria temporal
        self.memory = ConversationBufferWindowMemory(
            k=5,  # Mantener últimas 5 interacciones
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Crear agente
        self.agent = self._create_agent()
        
        # Crear executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=3,
            handle_parsing_errors=True,
            early_stopping_method="generate"
        )
    
    def _create_agent(self):
        """Crea el agente ReactAgent con prompt optimizado"""
        prompt_template = """Eres un asistente académico experto especializado en análisis de documentos IEEE y papers de investigación.

HERRAMIENTAS DISPONIBLES:
{tools}

NOMBRES DE HERRAMIENTAS: {tool_names}

INSTRUCCIONES ESPECÍFICAS:
1. 🔍 Para preguntas sobre contenido académico específico → usa "search_documents"
2. 🌐 Para información actual/noticias → usa "search_web" SOLO si se solicita explícitamente
3. 💬 Para saludos y preguntas generales → usa "direct_answer"
4. 📚 SIEMPRE cita las fuentes cuando uses search_documents
5. 🎯 Si no encuentras información específica, sugiere reformular la pregunta

FORMATO DE RESPUESTA OBLIGATORIO:
Thought: [Analiza qué herramienta necesitas usar y por qué]
Action: [nombre_exacto_de_herramienta]
Action Input: [texto_para_la_herramienta]
Observation: [resultado de la herramienta]
... (repite Thought/Action/Action Input/Observation si necesitas más herramientas)
Thought: I now know the final answer
Final Answer: [respuesta final completa y útil para el usuario]

CONTEXTO DE CONVERSACIÓN:
{chat_history}

PREGUNTA DEL USUARIO: {input}

Comienza tu análisis:
{agent_scratchpad}"""
        
        prompt = PromptTemplate.from_template(prompt_template)
        
        return create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
    
    def chat(self, message: str, session_id: str = "default") -> str:
        """Interfaz principal de chat"""
        try:
            # Ejecutar agente
            response = self.agent_executor.invoke({
                "input": message,
                "chat_history": self.memory.chat_memory.messages[-10:]  # Últimos 10 mensajes
            })
            
            # Extraer respuesta
            if isinstance(response, dict) and "output" in response:
                answer = response["output"]
            else:
                answer = str(response)
            
            # Limpiar respuesta si es necesario
            if "Final Answer:" in answer:
                answer = answer.split("Final Answer:")[-1].strip()
            
            return answer
            
        except Exception as e:
            error_msg = f"Error procesando consulta: {str(e)}"
            print(f"🔧 Debug - {error_msg}")
            
            # Respuesta de fallback más útil
            return """🤔 Disculpa, tuve un problema técnico procesando tu consulta. 

💡 **Puedes intentar**:
- Reformular tu pregunta de manera más específica
- Verificar que hay documentos cargados en el sistema
- Hacer una pregunta más simple primero

🔧 **O pregúntame sobre**:
- Contenido específico de tus documentos
- Metodologías mencionadas en los papers
- Resultados experimentales
- Autores y referencias

¿Te gustaría intentar con una pregunta diferente?"""
    
    def clear_memory(self):
        """Limpia la memoria de conversación"""
        self.memory.clear()
        print("🧹 Memoria de conversación limpiada")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Obtiene historial de conversación"""
        history = []
        for message in self.memory.chat_memory.messages:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})
        
        return history
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Estadísticas del agente"""
        return {
            "tools_available": len(self.tools),
            "tool_names": [tool.name for tool in self.tools],
            "memory_size": len(self.memory.chat_memory.messages),
            "model_name": self.llm.model,
            "temperature": self.llm.temperature
        }

def check_ollama_connection(model_name: str = "llama3.2:3b") -> bool:
    """Verifica que Ollama esté funcionando"""
    try:
        # Verificar si Ollama está corriendo
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if model_name in model_names:
                print(f"✅ Ollama conectado. Modelo {model_name} disponible.")
                return True
            else:
                print(f"❌ Modelo {model_name} no encontrado.")
                print(f"📋 Modelos disponibles: {model_names}")
                print(f"💡 Para instalar: ollama pull {model_name}")
                return False
        else:
            print("❌ Ollama no está respondiendo en puerto 11434")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ No se puede conectar con Ollama")
        print("🔧 Verifica que Ollama esté instalado y ejecutándose:")
        print("1. Instala desde: https://ollama.com/download")
        print("2. Ejecuta: ollama serve")
        print(f"3. Instala modelo: ollama pull {model_name}")
        return False
    except Exception as e:
        print(f"❌ Error verificando Ollama: {e}")
        return False

def test_agent_tools():
    """Función de prueba para verificar que las herramientas funcionan"""
    print("🧪 Probando definición de herramientas...")
    
    try:
        # Test básico de creación de herramientas
        rag_tool = RAGTool(rag_system=None)
        web_tool = WebSearchTool()
        direct_tool = DirectAnswerTool()
        
        print(f"✅ RAGTool: {rag_tool.name}")
        print(f"✅ WebSearchTool: {web_tool.name}")
        print(f"✅ DirectAnswerTool: {direct_tool.name}")
        print("✅ Todas las herramientas creadas correctamente")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en herramientas: {e}")
        return False

if __name__ == "__main__":
    # Test básico
    test_agent_tools()
# test_basic.py
from components.rag_system import RAGSystem
from components.agent_controller import AgentController

# Test básico del sistema
def test_system():
    # Inicializar RAG
    rag = RAGSystem()
    
    # Test de documentos de ejemplo
    test_docs = ["./data/documentos/ejemplo.pdf"]  # Añade tu PDF aquí
    
    if os.path.exists(test_docs[0]):
        print("🔄 Construyendo índice...")
        result = rag.build_index(test_docs)
        print(f"✅ Índice creado: {result}")
        
        # Inicializar agente
        agent = AgentController(rag)
        
        # Test de consultas
        test_queries = [
            "¿Qué metodologías se proponen?",
            "¿Cuáles son los resultados principales?",
            "¿Quiénes son los autores?",
        ]
        
        for query in test_queries:
            print(f"\n📝 Pregunta: {query}")
            response = agent.chat(query)
            print(f"🤖 Respuesta: {response}")
    else:
        print("❌ No se encontró archivo de prueba")

if __name__ == "__main__":
    test_system()
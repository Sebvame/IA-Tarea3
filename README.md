# IA-Tarea3
Plan de Implementación Completo: Asistente Conversacional RAG con Agentes


Arquitectura del Sistema:
┌─────────────────────────────────────────────────────────┐
│                    USUARIO                              │
│                    (Streamlit)                          │
└─────────────────────┬───────────────────────────────────┘
                      │ Pregunta
                      ▼
┌─────────────────────────────────────────────────────────┐
│               AGENTE ORQUESTADOR                        │
│              (Ollama + LangChain)                       │
│    ┌─────────────┐ ┌──────────────┐ ┌─────────────┐     │
│    │ Herramienta │ │ Herramienta  │ │ Herramienta │     │
│    │     RAG     │ │ Búsqueda Web │ │ Respuesta   │     │
│    │   (FAISS)   │ │  (Tavily)    │ │  Directa    │     │
│    └─────────────┘ └──────────────┘ └─────────────┘     │
└─────────────────────┬───────────────────────────────────┘
                      │ Respuesta
                      ▼
┌─────────────────────────────────────────────────────────┐
│                BASE DE DOCUMENTOS                       │
│        (Embeddings + Índice FAISS)                      │
└─────────────────────────────────────────────────────────┘

Estructura del proyecto:
rag_asistente/
├── requirements.txt
├── .env
├── app.py
├── components/
│   ├── __init__.py
│   ├── rag_system.py
│   ├── agent_controller.py
│   └── document_processor.py
└── utils/
    ├── __init__.py
    └── config.py


Para ejecutar:
    streamlit run app.py
# utils/config.py
import os
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """Carga configuración del sistema"""
    return {
        # Modelos
        'embedding_model': 'paraphrase-multilingual-MiniLM-L12-v2',
        'llm_model': 'llama3.2:3b',
        
        # RAG
        'chunk_size': 1200,
        'chunk_overlap': 300,
        'retrieval_k': 5,
        
        # Paths
        'data_path': './data',
        'indices_path': './data/indices',
        'documents_path': './data/documentos',
        
        # UI
        'max_file_size': 10 * 1024 * 1024,  # 10MB
        'allowed_extensions': ['.pdf'],
        
        # Agent
        'max_iterations': 3,
        'memory_window': 5,
        'temperature': 0.1,
    }

def ensure_directories():
    """Crea directorios necesarios"""
    config = load_config()
    
    directories = [
        config['data_path'],
        config['indices_path'],
        config['documents_path']
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
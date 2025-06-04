"""
Componentes principales del sistema RAG
"""

# Importaciones principales para facilitar el uso
from .rag_system import RAGSystem
from .agent_controller import AgentController
from .document_processor import DocumentProcessor

# Versión del paquete
__version__ = "1.0.0"

# Lista de componentes exportados
__all__ = [
    'RAGSystem',
    'AgentController', 
    'DocumentProcessor'
]

# Configuración de logging para debugging
import logging

# Configurar logging básico para el paquete
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Logger específico para este paquete
logger = logging.getLogger(__name__)
logger.info("Componentes RAG inicializados correctamente")
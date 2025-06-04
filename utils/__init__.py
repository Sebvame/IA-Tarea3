"""
Utilidades y configuraciones del sistema
"""

from .config import load_config, ensure_directories

__version__ = "1.0.0"

__all__ = [
    'load_config',
    'ensure_directories'
]

# Inicialización automática de directorios
try:
    ensure_directories()
    print("✅ Directorios del sistema verificados")
except Exception as e:
    print(f"⚠️ Advertencia al crear directorios: {e}")
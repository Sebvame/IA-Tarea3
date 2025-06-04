# components/document_processor.py
import re
import unicodedata
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import PyPDF2
import os
from datetime import datetime

class DocumentProcessor:
    def __init__(self, embedding_model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.embedding_model = SentenceTransformer(embedding_model_name)
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extrae metadata de documentos IEEE"""
        metadata = {
            'file_name': os.path.basename(pdf_path),
            'file_size': os.path.getsize(pdf_path),
            'processed_date': datetime.now().isoformat()
        }
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Información básica
                metadata['total_pages'] = len(reader.pages)
                
                # Extraer texto de primera página para detectar IEEE
                first_page = reader.pages[0].extract_text()
                
                # Detectar si es documento IEEE
                if 'IEEE' in first_page.upper():
                    metadata['document_type'] = 'IEEE_paper'
                    
                    # Extraer título (primera línea significativa)
                    lines = first_page.split('\n')
                    for line in lines:
                        if len(line.strip()) > 10 and not line.isupper():
                            metadata['title'] = line.strip()
                            break
                    
                    # Buscar DOI
                    doi_match = re.search(r'DOI[:\s]+(10\.\d+/[^\s]+)', first_page)
                    if doi_match:
                        metadata['doi'] = doi_match.group(1)
                    
                    # Buscar autores (típicamente después del título)
                    author_patterns = [
                        r'([A-Z][a-z]+ [A-Z][a-z]+(?:, [A-Z][a-z]+ [A-Z][a-z]+)*)',
                        r'([A-Z]\. [A-Z][a-z]+(?:, [A-Z]\. [A-Z][a-z]+)*)'
                    ]
                    for pattern in author_patterns:
                        authors = re.findall(pattern, first_page)
                        if authors:
                            metadata['authors'] = authors[0]
                            break
        
        except Exception as e:
            print(f"Error extracting metadata from {pdf_path}: {e}")
        
        return metadata
    
    def preprocess_text(self, text: str) -> str:
        """Preprocesamiento especializado para documentos académicos"""
        # Normalizar Unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Limpiar saltos de línea excesivos
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Corregir espaciado en referencias
        text = re.sub(r'\[\s*(\d+)\s*\]', r'[\1]', text)
        
        # Normalizar espacios múltiples
        text = re.sub(r' {2,}', ' ', text)
        
        # Mantener ecuaciones matemáticas
        text = re.sub(r'\$\s+([^$]+)\s+\$', r'$\1$', text)
        
        return text.strip()
    
    def semantic_chunking(self, text: str, chunk_size: int = 1200, overlap: int = 300) -> List[Dict]:
        """Chunking semántico para documentos IEEE"""
        # Detectar secciones típicas de papers IEEE
        ieee_sections = [
            r"ABSTRACT",
            r"I\.\s+INTRODUCTION",
            r"II\.\s+RELATED WORK",
            r"III\.\s+METHODOLOGY",
            r"IV\.\s+EXPERIMENTS?",
            r"V\.\s+RESULTS?",
            r"VI\.\s+CONCLUSION",
            r"REFERENCES"
        ]
        
        chunks = []
        current_section = "UNKNOWN"
        current_chunk = ""
        
        for line in text.split('\n'):
            # Detectar nueva sección
            for pattern in ieee_sections:
                if re.match(pattern, line.strip(), re.IGNORECASE):
                    # Guardar chunk anterior si existe
                    if current_chunk.strip():
                        chunks.append({
                            'text': current_chunk.strip(),
                            'section': current_section,
                            'chunk_size': len(current_chunk)
                        })
                    
                    current_section = line.strip()
                    current_chunk = ""
                    break
            
            current_chunk += line + "\n"
            
            # Controlar tamaño máximo
            if len(current_chunk) > chunk_size:
                chunks.append({
                    'text': current_chunk.strip(),
                    'section': current_section,
                    'chunk_size': len(current_chunk)
                })
                current_chunk = ""
        
        # Añadir último chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'section': current_section,
                'chunk_size': len(current_chunk)
            })
        
        return chunks
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Procesa completamente un PDF académico"""
        # Extraer metadata
        metadata = self.extract_metadata(pdf_path)
        
        # Extraer texto
        full_text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    full_text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return None
        
        # Preprocesar texto
        clean_text = self.preprocess_text(full_text)
        
        # Chunking semántico
        chunks = self.semantic_chunking(clean_text)
        
        # Enriquecer chunks con metadata
        for chunk in chunks:
            chunk.update(metadata)
            chunk['source'] = pdf_path
        
        return {
            'metadata': metadata,
            'chunks': chunks,
            'full_text': clean_text
        }
# components/rag_system.py
import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from components.document_processor import DocumentProcessor

class RAGSystem:
    def __init__(self, embedding_model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.document_processor = DocumentProcessor(embedding_model_name)
        self.index = None
        self.chunks = []
        self.metadata = []
        
    def build_index(self, pdf_paths: List[str], save_path: str = "./data/indices"):
        """Construye índice FAISS desde documentos PDF""" 
        print("🔄 Procesando documentos...")
        
        all_chunks = []
        all_metadata = []
        
        # Procesar cada PDF
        for pdf_path in pdf_paths:
            print(f"📄 Procesando: {os.path.basename(pdf_path)}")
            result = self.document_processor.process_pdf(pdf_path)
            
            if result:
                for chunk in result['chunks']:
                    all_chunks.append(chunk['text'])
                    all_metadata.append(chunk)
        
        if not all_chunks:
            raise ValueError("No se pudieron procesar documentos")
        
        print(f"📊 Generando embeddings para {len(all_chunks)} chunks...")
        
        # Generar embeddings
        embeddings = self.embedding_model.encode(
            all_chunks, 
            show_progress_bar=True,
            batch_size=32
        )
        
        # Crear índice FAISS 
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Almacenar datos
        self.chunks = all_chunks
        self.metadata = all_metadata
        
        # Guardar índice
        os.makedirs(save_path, exist_ok=True)
        
        # Guardar índice FAISS 
        faiss.write_index(self.index, f"{save_path}/faiss_index.index")
        
        # Guardar chunks y metadata
        with open(f"{save_path}/chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        
        with open(f"{save_path}/metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"✅ Índice creado con {len(all_chunks)} chunks y guardado en {save_path}")
        
        return {
            'total_chunks': len(all_chunks),
            'total_documents': len(pdf_paths),
            'index_path': save_path
        }
    
    def load_index(self, load_path: str = "./data/indices"):
        """Carga índice FAISS previamente guardado""" 
        try:
            # Cargar índice FAISS
            self.index = faiss.read_index(f"{load_path}/faiss_index.index")
            
            # Cargar chunks y metadata
            with open(f"{load_path}/chunks.pkl", 'rb') as f:
                self.chunks = pickle.load(f)
            
            with open(f"{load_path}/metadata.pkl", 'rb') as f:
                self.metadata = pickle.load(f)
            
            print(f"✅ Índice cargado: {len(self.chunks)} chunks")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando índice: {e}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Búsqueda en el índice FAISS""" 
        if self.index is None:
            raise ValueError("Índice no inicializado. Usa build_index() o load_index()")
        
        # Generar embedding de la consulta
        query_embedding = self.embedding_model.encode([query])
        
        # Buscar en FAISS 
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            min(k, len(self.chunks))
        )
        
        # Preparar resultados
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):  # Verificar índice válido
                similarity_score = 1 / (1 + distance)  # Convertir distancia a similaridad
                
                results.append({
                    'content': self.chunks[idx],
                    'metadata': self.metadata[idx],
                    'similarity_score': float(similarity_score),
                    'rank': i + 1
                })
        
        return results
    
    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.7) -> List[Dict[str, Any]]:
        """Búsqueda híbrida: semántica + palabras clave"""
        from rank_bm25 import BM25Okapi
        
        # Búsqueda semántica
        semantic_results = self.search(query, k=k*2)
        
        # Búsqueda BM25 (palabras clave)
        tokenized_chunks = [chunk.split() for chunk in self.chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        bm25_scores = bm25.get_scores(query.split())
        
        # Combinar scores
        combined_scores = {}
        
        # Scores semánticos
        for result in semantic_results:
            idx = self.chunks.index(result['content'])
            combined_scores[idx] = alpha * result['similarity_score']
        
        # Scores BM25 normalizados
        if len(bm25_scores) > 0:
            max_bm25 = max(bm25_scores)
            min_bm25 = min(bm25_scores)
            
            if max_bm25 > min_bm25:
                normalized_bm25 = [(score - min_bm25) / (max_bm25 - min_bm25) 
                                 for score in bm25_scores]
                
                for i, score in enumerate(normalized_bm25):
                    if i in combined_scores:
                        combined_scores[i] += (1 - alpha) * score
                    else:
                        combined_scores[i] = (1 - alpha) * score
        
        # Ordenar por score combinado y retornar top-k
        sorted_indices = sorted(combined_scores.items(), 
                              key=lambda x: x[1], reverse=True)[:k]
        
        final_results = []
        for idx, score in sorted_indices:
            final_results.append({
                'content': self.chunks[idx],
                'metadata': self.metadata[idx],
                'similarity_score': float(score),
                'rank': len(final_results) + 1
            })
        
        return final_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Estadísticas del sistema RAG"""
        if not self.chunks:
            return {"error": "No hay datos cargados"}
        
        # Análisis por sección
        sections = {}
        authors = set()
        document_types = set()
        
        for meta in self.metadata:
            section = meta.get('section', 'UNKNOWN')
            sections[section] = sections.get(section, 0) + 1
            
            if 'authors' in meta:
                authors.add(meta['authors'])
            
            if 'document_type' in meta:
                document_types.add(meta['document_type'])
        
        return {
            'total_chunks': len(self.chunks),
            'unique_documents': len(set(meta.get('file_name', '') for meta in self.metadata)),
            'sections_distribution': sections,
            'unique_authors': len(authors),
            'document_types': list(document_types),
            'average_chunk_size': sum(len(chunk) for chunk in self.chunks) / len(self.chunks),
            'index_dimension': self.index.d if self.index else 0
        }
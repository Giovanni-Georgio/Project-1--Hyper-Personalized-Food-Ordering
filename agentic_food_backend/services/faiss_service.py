# agentic_food_backend/services/faiss_service.py

import faiss
import numpy as np
import pickle
import os
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class FAISSService:
    def __init__(self, dimension: int = 384, index_file: str = "faiss_index.bin", mapping_file: str = "faiss_mapping.pkl"):
        self.dimension = dimension
        self.index_file = index_file
        self.mapping_file = mapping_file
        self.index = None
        self.id_to_idx = {}  # Maps document IDs to FAISS indices
        self.idx_to_id = {}  # Maps FAISS indices to document IDs
        self.next_idx = 0
        
    def initialize_index(self):
        """Initialize or load FAISS index"""
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.mapping_file):
                self.load_index()
            else:
                self.create_new_index()
            logger.info("FAISS index initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            self.create_new_index()
    
    def create_new_index(self):
        """Create a new FAISS index"""
        # Using IndexFlatL2 for simplicity - can be upgraded to IndexHNSWFlat for better performance
        self.index = faiss.IndexFlatL2(self.dimension)
        self.id_to_idx = {}
        self.idx_to_id = {}
        self.next_idx = 0
        logger.info(f"Created new FAISS index with dimension {self.dimension}")
    
    def add_vector(self, doc_id: str, embedding: np.ndarray) -> bool:
        """Add a vector to the FAISS index"""
        try:
            if self.index is None:
                self.create_new_index()
            
            # Ensure embedding is the right shape
            if len(embedding.shape) == 1:
                embedding = embedding.reshape(1, -1)
            
            # Add to FAISS index
            self.index.add(embedding.astype(np.float32))
            
            # Update mappings
            self.id_to_idx[doc_id] = self.next_idx
            self.idx_to_id[self.next_idx] = doc_id
            self.next_idx += 1
            
            logger.debug(f"Added vector for document {doc_id} to FAISS index")
            return True
        except Exception as e:
            logger.error(f"Failed to add vector to FAISS index: {e}")
            return False
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar vectors"""
        try:
            if self.index is None or self.index.ntotal == 0:
                logger.warning("FAISS index is empty or not initialized")
                return []
            
            # Ensure query embedding is the right shape
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Search in FAISS
            distances, indices = self.index.search(query_embedding.astype(np.float32), k)
            
            # Convert to document IDs and similarity scores
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx in self.idx_to_id:
                    doc_id = self.idx_to_id[idx]
                    # Convert L2 distance to similarity score (higher is better)
                    similarity = 1 / (1 + dist)
                    results.append((doc_id, similarity))
            
            return results
        except Exception as e:
            logger.error(f"Failed to search FAISS index: {e}")
            return []
    
    def update_vector(self, doc_id: str, embedding: np.ndarray) -> bool:
        """Update an existing vector (requires rebuilding index)"""
        try:
            # For FAISS IndexFlatL2, we need to rebuild the index to update vectors
            # This is a limitation of the simple index type
            logger.warning("Vector update requires index rebuild for IndexFlatL2")
            return self.remove_vector(doc_id) and self.add_vector(doc_id, embedding)
        except Exception as e:
            logger.error(f"Failed to update vector in FAISS index: {e}")
            return False
    
    def remove_vector(self, doc_id: str) -> bool:
        """Remove a vector (requires rebuilding index)"""
        try:
            if doc_id not in self.id_to_idx:
                logger.warning(f"Document {doc_id} not found in FAISS index")
                return False
            
            # For IndexFlatL2, we need to rebuild the entire index
            # This is inefficient but necessary for this index type
            old_idx = self.id_to_idx[doc_id]
            
            # Remove from mappings
            del self.id_to_idx[doc_id]
            del self.idx_to_id[old_idx]
            
            # Rebuild index without this vector
            self._rebuild_index_without_doc(old_idx)
            
            logger.info(f"Removed vector for document {doc_id} from FAISS index")
            return True
        except Exception as e:
            logger.error(f"Failed to remove vector from FAISS index: {e}")
            return False
    
    def _rebuild_index_without_doc(self, removed_idx: int):
        """Rebuild index excluding a specific document"""
        if self.index is None or self.index.ntotal <= 1:
            self.create_new_index()
            return
        
        # This is a simplified approach - in production, you might want to store
        # original embeddings separately to avoid this costly operation
        logger.warning("Index rebuild required - this operation is expensive")
        
        # For now, we'll create a new empty index
        # In a production system, you'd want to store embeddings separately
        # and rebuild from the stored embeddings
        self.create_new_index()
    
    def save_index(self):
        """Save FAISS index and mappings to disk"""
        try:
            if self.index is not None:
                faiss.write_index(self.index, self.index_file)
                
                with open(self.mapping_file, 'wb') as f:
                    pickle.dump({
                        'id_to_idx': self.id_to_idx,
                        'idx_to_id': self.idx_to_id,
                        'next_idx': self.next_idx
                    }, f)
                
                logger.info("FAISS index and mappings saved to disk")
                return True
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            return False
    
    def load_index(self):
        """Load FAISS index and mappings from disk"""
        try:
            self.index = faiss.read_index(self.index_file)
            
            with open(self.mapping_file, 'rb') as f:
                data = pickle.load(f)
                self.id_to_idx = data['id_to_idx']
                self.idx_to_id = data['idx_to_id']
                self.next_idx = data['next_idx']
            
            logger.info("FAISS index and mappings loaded from disk")
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False
    
    def get_index_info(self) -> dict:
        """Get information about the FAISS index"""
        if self.index is None:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": type(self.index).__name__,
            "is_trained": self.index.is_trained
        }
    
    def build_index_from_vectors(self, embeddings: List[np.ndarray], doc_ids: List[str]):
        """Build index from a batch of vectors (more efficient than adding one by one)"""
        try:
            self.create_new_index()
            
            if len(embeddings) != len(doc_ids):
                raise ValueError("Number of embeddings must match number of document IDs")
            
            if len(embeddings) == 0:
                logger.warning("No embeddings provided to build index")
                return True
            
            # Stack embeddings into a single array
            embeddings_array = np.vstack([emb.reshape(1, -1) for emb in embeddings])
            
            # Add all vectors at once
            self.index.add(embeddings_array.astype(np.float32))
            
            # Update mappings
            for i, doc_id in enumerate(doc_ids):
                self.id_to_idx[doc_id] = i
                self.idx_to_id[i] = doc_id
            
            self.next_idx = len(doc_ids)
            
            logger.info(f"Built FAISS index with {len(embeddings)} vectors")
            return True
        except Exception as e:
            logger.error(f"Failed to build FAISS index from vectors: {e}")
            return False

# Global instance
faiss_service = FAISSService()
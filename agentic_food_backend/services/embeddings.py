# Updated enhanced embeddings service with integrated FAISS and ChromaDB
# agentic_food_backend/services/embeddings.py

from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from agentic_food_backend.config import EMBEDDING_MODEL_NAME
from agentic_food_backend.services.chromadb_service import chromadb_service
from agentic_food_backend.services.faiss_service import faiss_service

logger = logging.getLogger(__name__)

class VectorMemorySystem:
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self.chromadb_service = chromadb_service
        self.faiss_service = faiss_service
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize ChromaDB and FAISS services"""
        try:
            # Initialize FAISS index
            self.faiss_service.initialize_index()
            logger.info("Vector memory system initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vector memory system: {e}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for given text with caching"""
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]
        
        try:
            embedding = self.embedding_model.encode([text])[0]
            self.embeddings_cache[text] = embedding
            
            # Limit cache size to prevent memory issues
            if len(self.embeddings_cache) > 1000:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self.embeddings_cache))
                del self.embeddings_cache[oldest_key]
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(384)  # Default dimension for all-MiniLM-L6-v2
    
    def add_menu_item_embedding(self, item_id: int, name: str, description: str, category: str):
        """Add menu item to both ChromaDB and FAISS vector stores"""
        try:
            # Create combined text for embedding
            combined_text = f"{name} {description or ''} {category or ''}".strip()
            embedding = self.generate_embedding(combined_text)
            
            doc_id = f"item_{item_id}"
            metadata = {"item_id": item_id, "type": "menu_item", "name": name, "category": category}
            
            # Add to ChromaDB
            success_chroma = self.chromadb_service.add_document(
                doc_id=doc_id,
                embedding=embedding.tolist(),
                document=combined_text,
                metadata=metadata
            )
            
            # Add to FAISS
            success_faiss = self.faiss_service.add_vector(doc_id, embedding)
            
            if success_chroma and success_faiss:
                logger.info(f"Added menu item {item_id} to vector stores")
            else:
                logger.warning(f"Partial failure adding menu item {item_id} to vector stores")
                
        except Exception as e:
            logger.error(f"Error adding menu item embedding: {e}")
    
    def add_user_preference_embedding(self, user_id: int, preferences: Dict):
        """Add user preferences to vector stores"""
        try:
            # Create text from preferences
            pref_text = " ".join([f"{k}:{v}" for k, v in preferences.items() if v])
            if not pref_text:
                logger.warning(f"Empty preferences for user {user_id}")
                return
            
            embedding = self.generate_embedding(pref_text)
            
            doc_id = f"user_{user_id}_pref"
            metadata = {"user_id": user_id, "type": "user_preference"}
            
            # Add to ChromaDB
            success_chroma = self.chromadb_service.add_document(
                doc_id=doc_id,
                embedding=embedding.tolist(),
                document=pref_text,
                metadata=metadata
            )
            
            # Add to FAISS
            success_faiss = self.faiss_service.add_vector(doc_id, embedding)
            
            if success_chroma and success_faiss:
                logger.info(f"Added user {user_id} preferences to vector stores")
            else:
                logger.warning(f"Partial failure adding user {user_id} preferences")
                
        except Exception as e:
            logger.error(f"Error adding user preference embedding: {e}")
    
    def find_similar_items(self, query: str, limit: int = 10, use_faiss: bool = True) -> List[Dict]:
        """Find similar menu items using FAISS (primary) or ChromaDB (fallback)"""
        try:
            query_embedding = self.generate_embedding(query)
            
            if use_faiss:
                # Try FAISS first (faster)
                faiss_results = self.faiss_service.search_similar(query_embedding, limit)
                
                if faiss_results:
                    similar_items = []
                    for doc_id, similarity in faiss_results:
                        if doc_id.startswith("item_"):
                            item_id = int(doc_id.split("_")[1])
                            similar_items.append({
                                "item_id": item_id,
                                "similarity_score": similarity,
                                "document": doc_id,
                                "source": "faiss"
                            })
                    
                    logger.debug(f"FAISS search returned {len(similar_items)} results")
                    return similar_items[:limit]
            
            # Fallback to ChromaDB
            chroma_results = self.chromadb_service.query_similar(
                query_embedding=query_embedding.tolist(),
                n_results=limit,
                where_filter={"type": "menu_item"}
            )
            
            similar_items = []
            if chroma_results and 'documents' in chroma_results:
                for i, (doc, meta, dist) in enumerate(zip(
                    chroma_results['documents'][0] if chroma_results['documents'] else [],
                    chroma_results['metadatas'][0] if chroma_results['metadatas'] else [],
                    chroma_results['distances'][0] if chroma_results['distances'] else []
                )):
                    similar_items.append({
                        "item_id": meta.get('item_id'),
                        "similarity_score": 1 - dist,  # Convert distance to similarity
                        "document": doc,
                        "source": "chroma"
                    })
            
            logger.debug(f"ChromaDB search returned {len(similar_items)} results")
            return similar_items
            
        except Exception as e:
            logger.error(f"Error finding similar items: {e}")
            return []
    
    def find_similar_preferences(self, user_id: int, limit: int = 5) -> List[Dict]:
        """Find users with similar preferences"""
        try:
            # Get user's preference embedding
            doc_id = f"user_{user_id}_pref"
            
            # Try to get user's embedding from FAISS first
            # This is a simplified approach - in production you'd want more sophisticated user similarity
            results = self.chromadb_service.query_similar(
                query_embedding=None,  # We'd need to get user's embedding first
                n_results=limit,
                where_filter={"type": "user_preference"}
            )
            
            # Process results similar to menu items
            similar_users = []
            # Implementation would depend on specific requirements
            
            return similar_users
            
        except Exception as e:
            logger.error(f"Error finding similar preferences: {e}")
            return []
    
    def update_menu_item_embedding(self, item_id: int, name: str, description: str, category: str):
        """Update menu item embedding in both stores"""
        try:
            combined_text = f"{name} {description or ''} {category or ''}".strip()
            embedding = self.generate_embedding(combined_text)
            
            doc_id = f"item_{item_id}"
            metadata = {"item_id": item_id, "type": "menu_item", "name": name, "category": category}
            
            # Update ChromaDB
            success_chroma = self.chromadb_service.update_document(
                doc_id=doc_id,
                embedding=embedding.tolist(),
                document=combined_text,
                metadata=metadata
            )
            
            # Update FAISS (requires rebuild for IndexFlatL2)
            success_faiss = self.faiss_service.update_vector(doc_id, embedding)
            
            if success_chroma or success_faiss:  # At least one succeeded
                logger.info(f"Updated menu item {item_id} in vector stores")
            else:
                logger.warning(f"Failed to update menu item {item_id} in vector stores")
                
        except Exception as e:
            logger.error(f"Error updating menu item embedding: {e}")
    
    def remove_menu_item_embedding(self, item_id: int):
        """Remove menu item from vector stores"""
        try:
            doc_id = f"item_{item_id}"
            
            # Remove from ChromaDB
            success_chroma = self.chromadb_service.delete_document(doc_id)
            
            # Remove from FAISS
            success_faiss = self.faiss_service.remove_vector(doc_id)
            
            if success_chroma or success_faiss:
                logger.info(f"Removed menu item {item_id} from vector stores")
            else:
                logger.warning(f"Failed to remove menu item {item_id} from vector stores")
                
        except Exception as e:
            logger.error(f"Error removing menu item embedding: {e}")
    
    def get_system_stats(self) -> Dict:
        """Get statistics about the vector memory system"""
        try:
            chroma_info = self.chromadb_service.get_collection_info()
            faiss_info = self.faiss_service.get_index_info()
            
            return {
                "chromadb": chroma_info,
                "faiss": faiss_info,
                "cache_size": len(self.embeddings_cache),
                "embedding_model": EMBEDDING_MODEL_NAME
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {"error": str(e)}
    
    def rebuild_faiss_from_chroma(self):
        """Rebuild FAISS index from ChromaDB data (maintenance operation)"""
        try:
            logger.info("Starting FAISS index rebuild from ChromaDB...")
            
            # Query all documents from ChromaDB
            all_results = self.chromadb_service.query_similar(
                query_embedding=[0] * 384,  # Dummy query to get all
                n_results=10000  # Large number to get all
            )
            
            if not all_results or not all_results.get('embeddings'):
                logger.warning("No data found in ChromaDB for rebuild")
                return False
            
            # Extract embeddings and IDs
            embeddings = [np.array(emb) for emb in all_results['embeddings'][0]]
            doc_ids = all_results['ids'][0]
            
            # Rebuild FAISS index
            success = self.faiss_service.build_index_from_vectors(embeddings, doc_ids)
            
            if success:
                logger.info("FAISS index rebuilt successfully")
                return True
            else:
                logger.error("Failed to rebuild FAISS index")
                return False
                
        except Exception as e:
            logger.error(f"Error rebuilding FAISS index: {e}")
            return False
    
    def save_state(self):
        """Save FAISS index to disk"""
        try:
            return self.faiss_service.save_index()
        except Exception as e:
            logger.error(f"Error saving vector memory state: {e}")
            return False
    
    def health_check(self) -> Dict:
        """Check health of vector memory components"""
        try:
            health = {
                "embedding_model": "loaded",
                "chromadb": "unknown",
                "faiss": "unknown",
                "cache": "active"
            }
            
            # Test embedding generation
            test_embedding = self.generate_embedding("test")
            if test_embedding is not None and len(test_embedding) > 0:
                health["embedding_model"] = "healthy"
            else:
                health["embedding_model"] = "failed"
            
            # Test ChromaDB
            chroma_info = self.chromadb_service.get_collection_info()
            if chroma_info.get("status") == "initialized":
                health["chromadb"] = "healthy"
            else:
                health["chromadb"] = "failed"
            
            # Test FAISS
            faiss_info = self.faiss_service.get_index_info()
            if faiss_info.get("status") == "initialized":
                health["faiss"] = "healthy"
            else:
                health["faiss"] = "failed"
            
            return health
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return {"error": str(e)}

# Global instance
vector_memory = VectorMemorySystem()
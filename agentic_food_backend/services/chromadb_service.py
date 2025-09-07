# agentic_food_backend/services/chromadb_service.py

import chromadb
from chromadb.config import Settings
import logging
from agentic_food_backend.config import CHROMA_DB_PATH

logger = logging.getLogger(__name__)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

async def initialize_chroma_collection():
    """Initialize ChromaDB collection on startup"""
    try:
        collection = chroma_client.get_or_create_collection(
            name="food_ordering_memory",
            metadata={"description": "User preferences and menu item embeddings"}
        )
        logger.info("ChromaDB collection initialized successfully")
        return collection
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB collection: {e}")
        return None

class ChromaDBService:
    def __init__(self):
        self.client = chroma_client
        self.collection = None
        
    async def initialize(self):
        """Initialize the collection"""
        self.collection = await initialize_chroma_collection()
        
    def add_document(self, doc_id: str, embedding: list, document: str, metadata: dict):
        """Add a document with embedding to ChromaDB"""
        if not self.collection:
            logger.warning("ChromaDB collection not initialized")
            return False
            
        try:
            self.collection.add(
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata],
                ids=[doc_id]
            )
            logger.info(f"Added document {doc_id} to ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Failed to add document to ChromaDB: {e}")
            return False
    
    def query_similar(self, query_embedding: list, n_results: int = 10, where_filter: dict = None):
        """Query for similar documents"""
        if not self.collection:
            logger.warning("ChromaDB collection not initialized")
            return []
            
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter
            )
            return results
        except Exception as e:
            logger.error(f"Failed to query ChromaDB: {e}")
            return []
    
    def update_document(self, doc_id: str, embedding: list, document: str, metadata: dict):
        """Update an existing document"""
        if not self.collection:
            logger.warning("ChromaDB collection not initialized")
            return False
            
        try:
            self.collection.update(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata]
            )
            logger.info(f"Updated document {doc_id} in ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Failed to update document in ChromaDB: {e}")
            return False
    
    def delete_document(self, doc_id: str):
        """Delete a document from ChromaDB"""
        if not self.collection:
            logger.warning("ChromaDB collection not initialized")
            return False
            
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted document {doc_id} from ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document from ChromaDB: {e}")
            return False
    
    def get_collection_info(self):
        """Get collection information"""
        if not self.collection:
            return {"status": "not_initialized"}
            
        try:
            count = self.collection.count()
            return {
                "status": "initialized",
                "document_count": count,
                "name": self.collection.name
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"status": "error", "error": str(e)}

# Global instance
chromadb_service = ChromaDBService()
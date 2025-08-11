"""
Milvus database handler for the Arabic-Audio-Preprocessing-and-Feature-Extraction
"""

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from config.config import Config

class MilvusHandler:
    """Handles all Milvus database operations"""
    
    def __init__(self, host=None, port=None):
        self.host = host or Config.DEFAULT_MILVUS_HOST
        self.port = port or Config.DEFAULT_MILVUS_PORT
        
        # Collection instances
        self.embedding_collection = None
        self.logmel_collection = None
        
        # Initialize connection and collections
        self.setup_connection()
        self.setup_collections()
    
    def setup_connection(self):
        """Initialize Milvus connection"""
        print("🔗 Setting up Milvus connection...")
        
        try:
            # Connect to Milvus
            connections.connect("default", host=self.host, port=self.port)
            print(f"✅ Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            print(f"❌ Failed to connect to Milvus: {str(e)}")
            print("Make sure Milvus is running on the specified host and port")
            raise
    
    def setup_collections(self):
        """Setup Milvus collections for embeddings and log-mel features"""
        # Speaker Embeddings Collection (512D for pyannote)
        embedding_fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="audio_name", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="speaker_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="audio_path", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="embedding_vector", dtype=DataType.FLOAT_VECTOR, dim=Config.EMBEDDING_DIM),
            FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=50)
        ]
        
        embedding_schema = CollectionSchema(embedding_fields, "Speaker embeddings collection")
        
        # Drop existing collection if it exists
        if utility.has_collection(Config.EMBEDDING_COLLECTION_NAME):
            utility.drop_collection(Config.EMBEDDING_COLLECTION_NAME)
        
        self.embedding_collection = Collection(Config.EMBEDDING_COLLECTION_NAME, embedding_schema)
        
        # Create index for embeddings
        self.embedding_collection.create_index("embedding_vector", Config.INDEX_PARAMS)
        
        # Log-Mel Features Collection (192D)
        logmel_fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="audio_name", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="speaker_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="audio_path", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="logmel_vector", dtype=DataType.FLOAT_VECTOR, dim=Config.LOGMEL_DIM),
            FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=50)
        ]
        
        logmel_schema = CollectionSchema(logmel_fields, "Log-mel features collection")
        
        # Drop existing collection if it exists
        if utility.has_collection(Config.LOGMEL_COLLECTION_NAME):
            utility.drop_collection(Config.LOGMEL_COLLECTION_NAME)
        
        self.logmel_collection = Collection(Config.LOGMEL_COLLECTION_NAME, logmel_schema)
        
        # Create index for log-mel features
        self.logmel_collection.create_index("logmel_vector", Config.INDEX_PARAMS)
        
        print("✅ Milvus collections created successfully")
        print(f"✅ Embedding dimension: {Config.EMBEDDING_DIM}D")
    
    def insert_data(self, embedding_data, logmel_data):
        """Insert embeddings and log-mel features to Milvus"""
        try:
            # Insert embedding data
            if embedding_data:
                self.embedding_collection.insert([embedding_data])
            
            # Insert log-mel data
            if logmel_data:
                self.logmel_collection.insert([logmel_data])
            
            return True
        except Exception as e:
            print(f"❌ Error inserting to Milvus: {str(e)}")
            return False
    
    def search_similar_speakers(self, query_embedding, top_k=5):
        """Search for similar speakers in Milvus"""
        try:
            # Load collection
            self.embedding_collection.load()
            
            # Perform search
            results = self.embedding_collection.search(
                data=[query_embedding],
                anns_field="embedding_vector",
                param=Config.SEARCH_PARAMS,
                limit=top_k,
                output_fields=["audio_name", "speaker_id", "audio_path"]
            )
            
            return results
        except Exception as e:
            print(f"❌ Error searching Milvus: {str(e)}")
            return None
    
    def flush_collections(self):
        """Flush data to Milvus"""
        try:
            self.embedding_collection.flush()
            self.logmel_collection.flush()
            print("💾 Data flushed to Milvus successfully")
            return True
        except Exception as e:
            print(f"⚠️ Warning: Error flushing to Milvus: {str(e)}")
            return False
    
    def get_collection_stats(self):
        """Get statistics about Milvus collections"""
        try:
            # Load collections
            self.embedding_collection.load()
            self.logmel_collection.load()
            
            # Get counts
            embedding_count = self.embedding_collection.num_entities
            logmel_count = self.logmel_collection.num_entities
            
            return embedding_count, logmel_count
        except Exception as e:
            print(f"❌ Error getting collection stats: {str(e)}")
            return 0, 0
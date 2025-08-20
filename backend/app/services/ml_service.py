import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import Tuple, List, Optional
import logging
from app.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLService:
    """Service for handling machine learning operations."""
    
    def __init__(self):
        """Initialize ML models."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize CLIP model for image-text embeddings
        try:
            self.clip_model = CLIPModel.from_pretrained(settings.CLIP_MODEL_NAME)
            self.clip_processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL_NAME)
            self.clip_model.to(self.device)
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
        
        # For now, we'll use a simple text embedding approach
        # In production, you'd want to use sentence-transformers
        logger.info("ML service initialized (CLIP only)")
    
    def process_image(self, image: Image.Image) -> np.ndarray:
        """
        Process image and extract CLIP embeddings.
        
        Args:
            image: PIL Image object
            
        Returns:
            numpy array of image embeddings
        """
        try:
            # Preprocess image for CLIP
            inputs = self.clip_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract image features
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = image_features.cpu().numpy()
            
            # Normalize embeddings
            image_features = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
            
            logger.info(f"Image processed successfully. Embedding shape: {image_features.shape}")
            return image_features
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise
    
    def process_text(self, text: str) -> np.ndarray:
        """
        Process text and extract CLIP text embeddings.
        
        Args:
            text: Input text string
            
        Returns:
            numpy array of text embeddings
        """
        try:
            if not text.strip():
                # Return zero vector for empty text
                return np.zeros((1, 512))  # CLIP embedding dimension
            
            # Use CLIP for text processing instead of sentence-transformers
            inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract text features
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
                text_features = text_features.cpu().numpy()
            
            # Normalize embeddings
            text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
            
            logger.info(f"Text processed successfully. Embedding shape: {text_features.shape}")
            return text_features
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            raise
    
    def calculate_similarity(self, query_embedding: np.ndarray, product_embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between query and product embeddings.
        
        Args:
            query_embedding: Query embedding vector
            product_embeddings: Matrix of product embeddings
            
        Returns:
            Array of similarity scores
        """
        try:
            # Calculate cosine similarity
            similarities = np.dot(product_embeddings, query_embedding.T).flatten()
            
            # Ensure scores are in [0, 1] range
            similarities = np.clip(similarities, 0, 1)
            
            logger.info(f"Similarity calculated for {len(similarities)} products")
            return similarities
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            raise
    
    def get_hybrid_embedding(self, image_embedding: np.ndarray, text_embedding: np.ndarray, 
                            image_weight: float = 0.7) -> np.ndarray:
        """
        Combine image and text embeddings for hybrid search.
        
        Args:
            image_embedding: Image embedding vector
            text_embedding: Text embedding vector
            image_weight: Weight for image embedding (0-1)
            
        Returns:
            Combined hybrid embedding
        """
        try:
            # Weighted combination of embeddings
            text_weight = 1 - image_weight
            
            # Ensure embeddings have same dimensions (pad if necessary)
            max_dim = max(image_embedding.shape[1], text_embedding.shape[1])
            
            if image_embedding.shape[1] < max_dim:
                image_embedding = np.pad(image_embedding, ((0, 0), (0, max_dim - image_embedding.shape[1])))
            if text_embedding.shape[1] < max_dim:
                text_embedding = np.pad(text_embedding, ((0, 0), (0, max_dim - text_embedding.shape[1])))
            
            # Combine embeddings
            hybrid_embedding = (image_weight * image_embedding + text_weight * text_embedding)
            
            # Normalize
            hybrid_embedding = hybrid_embedding / np.linalg.norm(hybrid_embedding, axis=1, keepdims=True)
            
            logger.info("Hybrid embedding created successfully")
            return hybrid_embedding
            
        except Exception as e:
            logger.error(f"Error creating hybrid embedding: {e}")
            raise


# Global ML service instance
ml_service = MLService()

import numpy as np
from typing import List, Optional, Tuple
import logging
from app.models.schemas import ProductRecommendation, ProductCategory
from app.services.ml_service import ml_service
from app.core.config import settings

logger = logging.getLogger(__name__)


class SearchService:
    """Service for handling product search operations."""
    
    def __init__(self):
        """Initialize search service with mock product database."""
        self.products = self._generate_mock_products()
        self.product_embeddings = self._generate_mock_embeddings()
        logger.info(f"Search service initialized with {len(self.products)} mock products")
        
        # Test the embeddings to ensure they're working correctly
        if settings.DEVELOPMENT_MODE:
            self.test_embeddings()
            self.debug_embeddings()
    
    def _generate_mock_products(self) -> List[dict]:
        """Generate mock product data for demonstration."""
        return [
            {
                "product_id": "prod_001",
                "name": "Nike Air Max 270 Running Shoes",
                "description": "Comfortable running shoes with Air Max cushioning technology",
                "category": ProductCategory.SHOES,
                "brand": "Nike",
                "price": 129.99,
                "image_url": "/static/images/nike.jpg",
                "embedding_type": "shoes_running"
            },
            {
                "product_id": "prod_002",
                "name": "Adidas Ultraboost 22",
                "description": "Premium running shoes with responsive Boost midsole",
                "category": ProductCategory.SHOES,
                "brand": "Adidas",
                "price": 189.99,
                "image_url": "/static/images/shoesAdidas.jpg",
                "embedding_type": "shoes_running"
            },
            {
                "product_id": "prod_003",
                "name": "Apple iPhone 15 Pro",
                "description": "Latest smartphone with advanced camera system and A17 Pro chip",
                "category": ProductCategory.ELECTRONICS,
                "brand": "Apple",
                "price": 999.99,
                "image_url": "/static/images/iphone.jpg",
                "embedding_type": "electronics_smartphone"
            },
            {
                "product_id": "prod_004",
                "name": "Samsung Galaxy S24",
                "description": "Android flagship with AI-powered features and excellent camera",
                "category": ProductCategory.ELECTRONICS,
                "brand": "Samsung",
                "price": 799.99,
                "image_url": "/static/images/samsung.jpg",
                "embedding_type": "electronics_smartphone"
            },
            {
                "product_id": "prod_005",
                "name": "Levi's 501 Original Jeans",
                "description": "Classic straight-leg jeans in authentic denim",
                "category": ProductCategory.CLOTHING,
                "brand": "Levi's",
                "price": 59.99,
                "image_url": "/static/images/nike.jpg",  # Using Nike image as placeholder
                "embedding_type": "clothing_jeans"
            },
            {
                "product_id": "prod_006",
                "name": "Uniqlo Cotton T-Shirt",
                "description": "Soft cotton t-shirt in various colors, perfect for everyday wear",
                "category": ProductCategory.CLOTHING,
                "brand": "Uniqlo",
                "price": 19.99,
                "image_url": "/static/images/nike.jpg",  # Using Nike image as placeholder
                "embedding_type": "clothing_tshirt"
            },
            {
                "product_id": "prod_007",
                "name": "Dyson V15 Detect Vacuum",
                "description": "Cordless vacuum with laser dust detection and powerful suction",
                "category": ProductCategory.HOME,
                "brand": "Dyson",
                "price": 699.99,
                "image_url": "/static/images/samsung.jpg",  # Using Samsung image as placeholder
                "embedding_type": "home_vacuum"
            },
            {
                "product_id": "prod_008",
                "name": "Lululemon Align Leggings",
                "description": "Buttery-soft yoga leggings with four-way stretch",
                "category": ProductCategory.CLOTHING,
                "brand": "Lululemon",
                "price": 98.00,
                "image_url": "/static/images/nike.jpg",  # Using Nike image as placeholder
                "embedding_type": "clothing_leggings"
            },
            {
                "product_id": "prod_009",
                "name": "Sony WH-1000XM5 Headphones",
                "description": "Premium noise-cancelling wireless headphones with exceptional sound",
                "category": ProductCategory.ELECTRONICS,
                "brand": "Sony",
                "price": 399.99,
                "image_url": "/static/images/samsung.jpg",  # Using Samsung image as placeholder
                "embedding_type": "electronics_headphones"
            },
            {
                "product_id": "prod_010",
                "name": "Patagonia Down Jacket",
                "description": "Lightweight insulated jacket perfect for cold weather adventures",
                "category": ProductCategory.CLOTHING,
                "brand": "Patagonia",
                "price": 229.00,
                "image_url": "/static/images/nike.jpg",  # Using Nike image as placeholder
                "embedding_type": "clothing_jacket"
            }
        ]
    
    def _generate_mock_embeddings(self) -> np.ndarray:
        """Generate real CLIP embeddings for mock products using their descriptions."""
        try:
            logger.info("Generating real CLIP embeddings for mock products...")
            
            embeddings = []
            for product in self.products:
                # Create a more descriptive text for each product
                product_text = f"{product['name']} {product['description']} {product['category']} {product['brand']}"
                logger.info(f"Processing: {product_text[:80]}...")
                
                # Use CLIP to generate real text embedding
                text_embedding = ml_service.process_text(product_text)
                
                # Ensure it's a 1D array and normalize
                if text_embedding.ndim > 1:
                    text_embedding = text_embedding.flatten()
                
                # Normalize the embedding
                norm = np.linalg.norm(text_embedding)
                if norm > 0:
                    text_embedding = text_embedding / norm
                else:
                    logger.warning(f"Zero norm embedding for {product['name']}, using random fallback")
                    text_embedding = np.random.randn(512)
                    text_embedding = text_embedding / np.linalg.norm(text_embedding)
                
                embeddings.append(text_embedding)
            
            embeddings_array = np.array(embeddings)
            logger.info(f"Generated real CLIP embeddings with shape: {embeddings_array.shape}")
            
            # Verify embeddings are properly normalized
            norms = np.linalg.norm(embeddings_array, axis=1)
            logger.info(f"Embedding norms - Min: {norms.min():.6f}, Max: {norms.max():.6f}, Mean: {norms.mean():.6f}")
            
            return embeddings_array
            
        except Exception as e:
            logger.error(f"Failed to generate real CLIP embeddings: {e}")
            logger.info("Falling back to mock embeddings for development...")
            return self._generate_fallback_embeddings()
    
    def _generate_fallback_embeddings(self) -> np.ndarray:
        """Fallback to simple mock embeddings if CLIP fails."""
        np.random.seed(42)
        num_products = len(self.products)
        embedding_dim = 512
        
        embeddings = np.random.randn(num_products, embedding_dim)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Simple clustering - much more conservative
        for i, product in enumerate(self.products):
            if "shoes" in product["embedding_type"]:
                embeddings[i] += np.random.normal(0, 0.01, embedding_dim)
            elif "electronics" in product["embedding_type"]:
                embeddings[i] += np.random.normal(0, 0.01, embedding_dim)
            elif "clothing" in product["embedding_type"]:
                embeddings[i] += np.random.normal(0, 0.01, embedding_dim)
        
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings
    
    def search_products(self, image_embedding: Optional[np.ndarray] = None, 
                       text_embedding: Optional[np.ndarray] = None,
                       category: Optional[ProductCategory] = None) -> List[ProductRecommendation]:
        """
        Search for products based on image and/or text embeddings.
        
        Args:
            image_embedding: Optional image embedding vector
            text_embedding: Optional text embedding vector
            category: Optional category filter
            
        Returns:
            List of product recommendations sorted by similarity
        """
        try:
            # Determine search type and create query embedding
            if image_embedding is not None and text_embedding is not None:
                query_embedding = ml_service.get_hybrid_embedding(image_embedding, text_embedding)
                search_type = "hybrid"
                logger.info("Performing hybrid search (image + text)")
            elif image_embedding is not None:
                query_embedding = image_embedding
                search_type = "image"
                logger.info("Performing image-only search")
            elif text_embedding is not None:
                query_embedding = text_embedding
                search_type = "text"
                logger.info("Performing text-only search")
            else:
                raise ValueError("At least one of image_embedding or text_embedding must be provided")
            
            logger.info(f"Query embedding shape: {query_embedding.shape}")
            logger.info(f"Query embedding norm: {np.linalg.norm(query_embedding):.3f}")
            
            # Calculate similarities
            similarities = ml_service.calculate_similarity(query_embedding, self.product_embeddings)
            
            logger.info(f"Similarity scores: {similarities}")
            logger.info(f"Similarity threshold: {settings.SIMILARITY_THRESHOLD}")
            
            # Validate that we have meaningful similarities
            max_similarity = np.max(similarities)
            min_similarity = np.min(similarities)
            mean_similarity = np.mean(similarities)
            logger.info(f"Similarity stats - Max: {max_similarity:.3f}, Min: {min_similarity:.3f}, Mean: {mean_similarity:.3f}")
            
            # If all similarities are very low, this indicates a problem
            if max_similarity < 0.1:
                logger.warning("All similarity scores are very low - this may indicate embedding issues")
            
            # Create product recommendations with scores
            recommendations = []
            for i, (product, similarity) in enumerate(zip(self.products, similarities)):
                # Apply category filter if specified
                if category and product["category"] != category:
                    logger.info(f"Skipping {product['name']} due to category filter: {product['category']} != {category}")
                    continue
                
                # Apply similarity threshold
                if similarity < settings.SIMILARITY_THRESHOLD:
                    logger.info(f"Skipping {product['name']} due to low similarity: {similarity:.3f} < {settings.SIMILARITY_THRESHOLD}")
                    # In development mode, show all results regardless of threshold
                    if not settings.DEVELOPMENT_MODE:
                        continue
                    else:
                        logger.info(f"Development mode: including {product['name']} despite low similarity")
                
                logger.info(f"Adding {product['name']} with similarity: {similarity:.3f}")
                
                recommendation = ProductRecommendation(
                    product_id=product["product_id"],
                    name=product["name"],
                    description=product["description"],
                    category=product["category"],
                    brand=product["brand"],
                    price=product["price"],
                    image_url=product["image_url"],
                    similarity_score=float(similarity),
                    match_type=search_type
                )
                recommendations.append(recommendation)
            
            # Sort by similarity score (descending)
            recommendations.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Limit to max recommendations
            recommendations = recommendations[:settings.MAX_RECOMMENDATIONS]
            
            logger.info(f"Found {len(recommendations)} recommendations with {search_type} search")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in product search: {e}")
            raise
    
    def get_search_metadata(self, search_type: str, total_results: int) -> dict:
        """Generate search metadata for response."""
        return {
            "search_type": search_type,
            "total_products_in_db": len(self.products),
            "similarity_threshold": settings.SIMILARITY_THRESHOLD,
            "max_recommendations": settings.MAX_RECOMMENDATIONS,
            "search_timestamp": str(np.datetime64('now'))
        }

    def test_embeddings(self):
        """Test function to verify embeddings are working correctly."""
        logger.info("Testing CLIP embeddings...")
        
        # Test multiple queries to verify different categories work
        test_queries = [
            "comfortable running shoes",
            "smartphone with camera", 
            "jeans pants",
            "vacuum cleaner"
        ]
        
        for test_query in test_queries:
            logger.info(f"\n--- Testing query: '{test_query}' ---")
            
            try:
                test_embedding = ml_service.process_text(test_query)
                
                # Ensure embedding is normalized
                if test_embedding.ndim > 1:
                    test_embedding = test_embedding.flatten()
                norm = np.linalg.norm(test_embedding)
                if norm > 0:
                    test_embedding = test_embedding / norm
                
                # Calculate similarities
                similarities = ml_service.calculate_similarity(test_embedding, self.product_embeddings)
                
                # Show top matches
                top_indices = np.argsort(similarities)[::-1][:5]
                logger.info(f"Top matches for '{test_query}':")
                for i, idx in enumerate(top_indices):
                    product = self.products[idx]
                    similarity = similarities[idx]
                    category = product['category']
                    logger.info(f"  {i+1}. {product['name']} ({category}): {similarity:.3f}")
                
                # Validate that results make sense
                if "shoes" in test_query.lower():
                    shoe_products = [i for i, p in enumerate(self.products) if "shoes" in p["embedding_type"]]
                    shoe_similarities = [similarities[i] for i in shoe_products]
                    if shoe_similarities:
                        max_shoe_sim = max(shoe_similarities)
                        logger.info(f"  Best shoe similarity: {max_shoe_sim:.3f}")
                        if max_shoe_sim < 0.3:
                            logger.warning(f"  ⚠️  Shoe similarity too low - embeddings may not be working correctly")
                
                if "smartphone" in test_query.lower() or "camera" in test_query.lower():
                    phone_products = [i for i, p in enumerate(self.products) if "electronics" in p["embedding_type"]]
                    phone_similarities = [similarities[i] for i in phone_products]
                    if phone_similarities:
                        max_phone_sim = max(phone_similarities)
                        logger.info(f"  Best phone similarity: {max_phone_sim:.3f}")
                        if max_phone_sim < 0.3:
                            logger.warning(f"  ⚠️  Phone similarity too low - embeddings may not be working correctly")
                
            except Exception as e:
                logger.error(f"Error testing query '{test_query}': {e}")
        
        return True

    def debug_embeddings(self):
        """Debug function to inspect embeddings manually."""
        logger.info("=== DEBUGGING EMBEDDINGS ===")
        
        # Check embedding shapes and norms
        logger.info(f"Product embeddings shape: {self.product_embeddings.shape}")
        norms = np.linalg.norm(self.product_embeddings, axis=1)
        logger.info(f"Embedding norms: {norms}")
        
        # Test a simple query manually
        test_text = "running shoes"
        logger.info(f"Testing manual query: '{test_text}'")
        
        try:
            # Generate query embedding
            query_embedding = ml_service.process_text(test_text)
            if query_embedding.ndim > 1:
                query_embedding = query_embedding.flatten()
            
            # Normalize
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm
            logger.info(f"Query embedding norm: {query_norm:.6f}")
            
            # Calculate similarities manually
            similarities = np.dot(self.product_embeddings, query_embedding)
            logger.info(f"Raw similarities: {similarities}")
            
            # Show which products should match
            for i, (product, sim) in enumerate(zip(self.products, similarities)):
                if "shoes" in product["embedding_type"]:
                    logger.info(f"  SHOES: {product['name']} - {sim:.3f}")
                elif "electronics" in product["embedding_type"]:
                    logger.info(f"  ELECTRONICS: {product['name']} - {sim:.3f}")
                else:
                    logger.info(f"  OTHER: {product['name']} - {sim:.3f}")
                    
        except Exception as e:
            logger.error(f"Error in debug_embeddings: {e}")
            import traceback
            logger.error(traceback.format_exc())


# Global search service instance
search_service = SearchService()

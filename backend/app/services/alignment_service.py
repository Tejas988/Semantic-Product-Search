import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Tuple, Optional
import logging
import google.generativeai as genai
from app.core.config import settings

logger = logging.getLogger(__name__)


class AlignmentService:
    """Service for checking alignment between image content and text description."""
    
    def __init__(self):
        """Initialize the alignment service."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize CLIP for image captioning
        try:
            self.clip_model = CLIPModel.from_pretrained(settings.CLIP_MODEL_NAME)
            self.clip_processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL_NAME)
            self.clip_model.to(self.device)
            logger.info("CLIP model loaded for alignment service")
        except Exception as e:
            logger.error(f"Failed to load CLIP model for alignment: {e}")
            raise
        
        # Initialize Gemini client for LLM calls
        try:
            if hasattr(settings, 'GEMINI_API_KEY') and settings.GEMINI_API_KEY:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self.gemini_available = True
                self.gemini_model = genai.GenerativeModel('gemini-pro')
                logger.info("Gemini client initialized")
            else:
                self.gemini_available = False
                logger.warning("Gemini API key not configured, LLM alignment checks will be disabled")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self.gemini_available = False
    
    def generate_image_caption(self, image: Image.Image) -> str:
        """
        Generate a descriptive caption for the image using CLIP zero-shot classification.
        
        Args:
            image: PIL Image object
            
        Returns:
            String caption describing the image content
        """
        try:
            # Define common product categories and attributes for zero-shot classification
            candidate_labels = [
                "mobile phone", "smartphone", "laptop", "computer", "tablet",
                "shoes", "sneakers", "running shoes", "athletic shoes", "casual shoes",
                "clothing", "shirt", "pants", "dress", "jacket", "coat",
                "accessories", "bag", "backpack", "wallet", "watch", "jewelry",
                "electronics", "camera", "headphones", "speaker", "gaming console",
                "furniture", "chair", "table", "sofa", "bed", "desk",
                "sports equipment", "bicycle", "tennis racket", "basketball", "soccer ball",
                "food", "beverage", "snack", "fruit", "vegetable",
                "book", "magazine", "newspaper", "document"
            ]
            
            # Preprocess image for CLIP
            inputs = self.clip_processor(images=image, text=candidate_labels, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get image-text similarity scores
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Get top 3 most likely labels
            top_indices = torch.topk(probs, 3, dim=1).indices[0]
            top_labels = [candidate_labels[idx] for idx in top_indices]
            top_probs = [probs[0][idx].item() for idx in top_indices]
            
            # Generate a natural language caption
            if top_probs[0] > 0.3:  # Confidence threshold
                caption = f"This image shows {top_labels[0]}"
                if top_probs[1] > 0.2:
                    caption += f" and possibly {top_labels[1]}"
                caption += "."
            else:
                caption = "This image shows various objects or products."
            
            logger.info(f"Generated caption: {caption}")
            return caption
            
        except Exception as e:
            logger.error(f"Error generating image caption: {e}")
            return "Unable to generate image caption"
    
    def check_alignment_with_llm(self, image_caption: str, user_description: str) -> Dict[str, any]:
        """
        Use LLM to check if the image content aligns with the user's description.
        
        Args:
            image_caption: Generated caption describing the image
            user_description: User's text description of what they want
            
        Returns:
            Dictionary with alignment results and reasoning
        """
        if not self.gemini_available:
            return {
                "aligned": True,
                "confidence": 0.8,
                "reasoning": "LLM alignment check unavailable, assuming alignment",
                "warning": "Gemini API not configured"
            }
        
        try:
            # Create prompt for alignment check
            prompt = f"""
            You are a product search assistant. Your task is to determine if a user's product description 
            makes sense given what's shown in an image.
            
            Image content: {image_caption}
            User's description: {user_description}
            
            Please analyze if the user's description aligns with the image content. Consider:
            1. Does the description match the type of product shown?
            2. Are the attributes mentioned (color, style, type) consistent with the image?
            3. Is this a logical search request?
            
            Respond with a JSON object containing:
            - "aligned": boolean (true if description makes sense for the image)
            - "confidence": float (0.0 to 1.0, how confident you are)
            - "reasoning": string (explanation of your decision)
            - "suggestions": string (helpful suggestions if not aligned)
            
            Only respond with valid JSON.
            """
            
            # Make Gemini API call
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Try to extract JSON from response
            try:
                import json
                # Find JSON content in the response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx]
                    result = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
                
                logger.info(f"LLM alignment check completed: {result}")
                return result
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse LLM response as JSON: {e}")
                # Fallback: analyze response manually
                response_lower = response_text.lower()
                aligned = any(word in response_lower for word in ['aligned', 'match', 'consistent', 'makes sense'])
                confidence = 0.7 if aligned else 0.3
                
                return {
                    "aligned": aligned,
                    "confidence": confidence,
                    "reasoning": f"LLM response: {response_text}",
                    "suggestions": "Please provide a more specific description if the current one doesn't match the image."
                }
                
        except Exception as e:
            logger.error(f"Error in LLM alignment check: {e}")
            return {
                "aligned": True,
                "confidence": 0.6,
                "reasoning": f"LLM check failed: {str(e)}",
                "warning": "Alignment check encountered an error"
            }
    
    def validate_image_description_alignment(self, image: Image.Image, user_description: str) -> Dict[str, any]:
        """
        Main method to validate if user description aligns with image content.
        
        Args:
            image: PIL Image object
            user_description: User's text description
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Generate image caption
            image_caption = self.generate_image_caption(image)
            
            # Check alignment using LLM
            alignment_result = self.check_alignment_with_llm(image_caption, user_description)
            
            # Add image caption to result
            result = {
                "image_caption": image_caption,
                "user_description": user_description,
                "alignment_check": alignment_result,
                "is_aligned": alignment_result.get("aligned", True),
                "confidence": alignment_result.get("confidence", 0.8)
            }
            
            logger.info(f"Alignment validation completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in alignment validation: {e}")
            return {
                "image_caption": "Error generating caption",
                "user_description": user_description,
                "alignment_check": {
                    "aligned": True,
                    "confidence": 0.5,
                    "reasoning": f"Validation failed: {str(e)}",
                    "warning": "Alignment validation encountered an error"
                },
                "is_aligned": True,
                "confidence": 0.5
            }


# Global alignment service instance
alignment_service = AlignmentService()

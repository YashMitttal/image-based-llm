"""
Module for performing image-based analysis using OpenAI's GPT-4O model.

This module provides a robust implementation for processing and analyzing images
using OpenAI's capabilities. It supports multiple image input formats,
maintains chat history, and offers configurable parameters.
"""

import os
import base64
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageInput(BaseModel):
    """Model for image input validation and processing."""
    path: Optional[str] = None
    base64_data: Optional[str] = None
    url: Optional[str] = None

    @validator('path')
    def validate_path(cls, v):
        if v and not os.path.exists(v):
            raise ValueError(f"Image file not found at path: {v}")
        return v

class OpenAIConfig(BaseModel):
    """Configuration model for OpenAI client."""
    api_key: str
    model: str = "gpt-4o"
    max_tokens: int = 300
    temperature: float = 1.0
    default_prompt: str = "What do you see on this image?"

class ImageAnalyzer:
    """
    Main class for handling image analysis using OpenAI's GPT-4O model.
    
    Attributes:
        config (OpenAIConfig): Configuration for OpenAI client
        client (OpenAI): OpenAI client instance
        chat_history (List[Dict]): History of chat messages
    """

    def __init__(self, config: Optional[OpenAIConfig] = None):
        """
        Initialize the ImageAnalyzer with optional configuration.

        Args:
            config (OpenAIConfig, optional): Configuration for OpenAI client.
                If not provided, loads from environment variables.
        """
        load_dotenv()
        
        if config is None:
            config = OpenAIConfig(
                api_key=os.getenv("OPENAI_API_TOKEN")
            )
        
        self.config = config
        self.client = OpenAI(api_key=config.api_key)
        self.chat_history: List[Dict] = []

    def _encode_image(self, image_path: str) -> str:
        """
        Encode an image file to base64 format.

        Args:
            image_path (str): Path to the image file

        Returns:
            str: Base64 encoded image data

        Raises:
            IOError: If there's an error reading the image file
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise IOError(f"Failed to encode image: {str(e)}")

    def analyze_image(
        self, 
        image_input: ImageInput, 
        prompt: Optional[str] = None,
        store_history: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze an image and generate a response.

        Args:
            image_input (ImageInput): Image data to analyze
            prompt (str, optional): Custom prompt for analysis. 
                If not provided, uses default prompt
            store_history (bool): Whether to store chat history

        Returns:
            Dict[str, Any]: Analysis response from the model

        Raises:
            ValueError: If no valid image data is provided
            Exception: For any other processing errors
        """
        try:
            # Use default prompt if none provided
            if prompt is None:
                prompt = self.config.default_prompt

            # Prepare image content
            if image_input.path:
                base64_image = self._encode_image(image_input.path)
                image_url = f"data:image/jpeg;base64,{base64_image}"
            elif image_input.base64_data:
                image_url = f"data:image/jpeg;base64,{image_input.base64_data}"
            elif image_input.url:
                image_url = image_input.url
            else:
                raise ValueError("No valid image data provided")

            # Prepare message content
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }

            if store_history:
                self.chat_history.append(message)

            # Create chat completion
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[message],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )

            # Store response in history if enabled
            if store_history:
                self.chat_history.append({
                    "role": "assistant",
                    "content": response.choices[0].message.content
                })

            return response.dict()

        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            raise

    def clear_history(self):
        """Clear the chat history."""
        self.chat_history = []

    def get_history(self) -> List[Dict]:
        """
        Get the current chat history.

        Returns:
            List[Dict]: List of chat messages
        """
        return self.chat_history

# Example usage
if __name__ == "__main__":
    # Initialize analyzer with default configuration
    analyzer = ImageAnalyzer()
    
    try:
        # Process an image with default prompt
        response = analyzer.analyze_image(
            ImageInput(path="images/sample_image2.jpg")
        )
        print(response["choices"][0]["message"]["content"])

        # Process an image with custom prompt
        response = analyzer.analyze_image(
            ImageInput(path="images/sample_image2.jpg"),
            prompt="Describe the main elements in this image"
        )
        print(response["choices"][0]["message"]["content"])

    except Exception as e:
        logger.error(f"Error in image analysis: {e}")

import os
from typing import Generator, List, Dict, Any
import groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GroqClient:
    def __init__(self):
        # Initialize the Groq client with API key from environment variables
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = groq.Client(api_key=api_key)
        self.model = os.getenv("GROQ_MODEL", "llama3-8b-8192")  # Default model
    
    def generate_response(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """
        Generate a streaming response from the Groq API
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Yields:
            Chunks of the generated response
        """
        try:
            # Create a streaming completion
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                max_tokens=4096,
            )
            
            # Yield each chunk of the response
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"Error: {str(e)}"
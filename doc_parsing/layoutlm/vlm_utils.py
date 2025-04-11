import base64
import io
from openai import OpenAI, OpenAIError
from PIL import Image
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Assuming config.py is in the same directory or accessible
try:
    import config
except ImportError:
    print("Warning: config.py not found. VLM features may not work correctly.")
    # Define fallback config or handle error appropriately
    # load_dotenv() # Removed redundant call here
    import os # Need os for getenv
    config = type('obj', (object,), {
        'OPENAI_API_KEY': os.getenv("OPENAI_API_KEY"), # Still try to get from env
        'VLM_PROMPT_DESCRIPTION': "Describe this image.",
        # Add other necessary fallback config values if needed
    })()

_openai_client = None

def get_openai_client():
    """Initializes and returns the OpenAI client."""
    global _openai_client
    if _openai_client is None:
        if config.OPENAI_API_KEY:
            _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
            print("OpenAI client initialized.")
        else:
            print("Warning: OpenAI API key not configured. Cannot initialize client.")
            # Return None or raise an error based on how critical VLM is
            return None
    return _openai_client

def encode_image_to_base64(image: Image.Image, format="JPEG") -> str:
    """Encodes a PIL Image to a base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8')

def analyze_image_region_with_vlm(
    image_region: Image.Image,
    prompt: str = config.VLM_PROMPT_DESCRIPTION,
    # model: str = config.GPT4_VLM_MODEL_NAME # Use if specific model needed
    model: str = "gpt-4o" # Updated model name (was gpt-4-vision-preview)
) -> Optional[Dict[str, Any]]:
    """
    Analyzes an image region using GPT-4 Vision API.

    Args:
        image_region: PIL Image object of the region to analyze.
        prompt: The text prompt to guide the VLM analysis.
        model: The specific VLM model to use.

    Returns:
        A dictionary containing the API response, or None if an error occurs
        or the client is not initialized.
    """
    client = get_openai_client()
    if not client:
        print("Error: OpenAI client not available.")
        return None

    try:
        base64_image = encode_image_to_base64(image_region)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=300 # Adjust max_tokens as needed
        )
        # Return the full response object or extract specific parts like response.choices[0].message.content
        return response.model_dump() # Return Pydantic model as dict

    except OpenAIError as e:
        print(f"Error calling OpenAI API: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during VLM analysis: {e}")
        return None

# Example usage (for testing purposes)
if __name__ == '__main__':
    print("VLM Utils - Example Usage (requires API key and test image)")
    # client = get_openai_client()
    # if client:
    #     try:
    #         # Create a dummy image for testing
    #         dummy_image = Image.new('RGB', (200, 100), color = 'red')
    #         print("Sending dummy image to VLM...")
    #         analysis = analyze_image_region_with_vlm(dummy_image, "What color is this image?")
    #         if analysis:
    #             print("VLM Analysis Result:")
    #             # Extracting the text content from the response
    #             if analysis.get('choices') and len(analysis['choices']) > 0:
    #                  message = analysis['choices'][0].get('message')
    #                  if message and message.get('content'):
    #                      print(message['content'])
    #                  else:
    #                      print("Could not extract content from VLM response.")
    #             else:
    #                 print("No choices found in VLM response.")
    #             # print(json.dumps(analysis, indent=2)) # Print full response
    #         else:
    #             print("VLM analysis failed.")
    #     except Exception as e:
    #         print(f"Error during VLM example usage: {e}")
    # else:
    #     print("Skipping VLM example usage as client could not be initialized.")

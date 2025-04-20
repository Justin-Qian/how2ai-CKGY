import base64
import io
# Use AsyncOpenAI for concurrent API calls
from openai import AsyncOpenAI, OpenAIError
from PIL import Image
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from . import config
import asyncio # Needed for async operations
import httpx # OpenAI async client uses httpx
import os # Import os here for path manipulation

# --- Robust Config Loading ---
# Get the directory where this script (vlm_utils.py) is located
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables from .env file in the script's directory
_ENV_PATH = os.path.join(_SCRIPT_DIR, '.env')
if os.path.exists(_ENV_PATH):
    load_dotenv(dotenv_path=_ENV_PATH)
else:
    # Also check parent directory (e.g., if utils are in a subdir)
    _PARENT_ENV_PATH = os.path.join(os.path.dirname(_SCRIPT_DIR), '.env')
    if os.path.exists(_PARENT_ENV_PATH):
         load_dotenv(dotenv_path=_PARENT_ENV_PATH)

# Attempt to import config relative to the script location
try:
    # This relative import should work when run as a module (python -m ...)
    from . import config 
except ImportError:
    # Fallback if run directly or structure is unexpected
    print("Warning: Relative import of config failed. Attempting fallback.")
    try:
        # Try importing assuming layoutlm is in PYTHONPATH
        import config 
    except ImportError:
        print("ERROR: config.py not found. VLM features will be disabled or use hardcoded defaults.")
        # Define fallback config object with essential defaults
        config = type('obj', (object,), {
            'OPENAI_API_KEY': os.getenv("OPENAI_API_KEY"), 
            'VLM_PROMPT_DESCRIPTION': "Describe this image.",
             # Add other essential fallbacks if needed
             'VLM_PROMPT_FIGURE': "Describe this figure from a research paper.",
             'VLM_PROMPT_DRAWING': "Describe this drawing or diagram.",
             'VLM_PROMPT_TABLE': "Summarize this table.",
             'VLM_PROMPT_EQUATION': "Transcribe this equation.",
        })()

_openai_client = None
_async_openai_client = None # Add async client variable

# Keep synchronous client function if needed elsewhere, but VLM will use async
def get_openai_client():
    """Initializes and returns the synchronous OpenAI client."""
    global _openai_client
    if _openai_client is None:
        if config.OPENAI_API_KEY:
            # Consider adding timeout configuration
            _openai_client = OpenAI(
                api_key=config.OPENAI_API_KEY,
                timeout=httpx.Timeout(30.0, connect=5.0) # Example timeout
            )
            print("Sync OpenAI client initialized.")
        else:
            print("Warning: OpenAI API key not configured. Cannot initialize sync client.")
            return None
    return _openai_client

# New function for async client
def get_async_openai_client():
    """Initializes and returns the asynchronous OpenAI client."""
    global _async_openai_client
    if _async_openai_client is None:
        if config.OPENAI_API_KEY:
             # Use httpx.AsyncClient for async timeouts
            timeout = httpx.Timeout(30.0, connect=5.0) # Example timeout (adjust as needed)
            _async_openai_client = AsyncOpenAI(
                api_key=config.OPENAI_API_KEY,
                timeout=timeout
                # Consider adding max_retries if needed
            )
            print("Async OpenAI client initialized.")
        else:
            print("Warning: OpenAI API key not configured. Cannot initialize async client.")
            return None
    return _async_openai_client

def encode_image_to_base64(image: Image.Image, format="JPEG") -> str:
    """Encodes a PIL Image to a base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8')

# Make the function asynchronous
async def analyze_image_region_with_vlm(
    image_region: Image.Image,
    prompt: str = config.VLM_PROMPT_DESCRIPTION,
    # model: str = config.GPT4_VLM_MODEL_NAME # Use if specific model needed
    model: str = "gpt-4o" # Updated model name (was gpt-4-vision-preview)
) -> Optional[Dict[str, Any]]:
    """
    Analyzes an image region using GPT-4 Vision API asynchronously.

    Args:
        image_region: PIL Image object of the region to analyze.
        prompt: The text prompt to guide the VLM analysis.
        model: The specific VLM model to use.

    Returns:
        A dictionary containing the API response, or None if an error occurs
        or the client is not initialized.
    """
    # Get the async client
    client = get_async_openai_client()
    if not client:
        print("Error: Async OpenAI client not available.")
        return None

    try:
        # Encoding needs to happen synchronously before the async call
        base64_image = encode_image_to_base64(image_region)

        # Use await for the async API call
        response = await client.chat.completions.create(
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
        return response.model_dump() # Return Pydantic model as dict

    except OpenAIError as e:
        print(f"Error calling OpenAI API (async): {e}")
        return None
    except Exception as e:
        # Catch potential timeout errors from httpx as well
        if isinstance(e, httpx.TimeoutException):
             print(f"OpenAI API call timed out: {e}")
        else:
             print(f"An unexpected error occurred during async VLM analysis: {e}")
        return None

# Example usage needs to be adapted for async
async def main_test():
    print("Async VLM Utils - Example Usage (requires API key and test image)")
    client = get_async_openai_client()
    if client:
        try:
            dummy_image = Image.new('RGB', (200, 100), color = 'red')
            print("Sending dummy image to VLM (async)...")
            # Await the async function call
            analysis = await analyze_image_region_with_vlm(dummy_image, "What color is this image?")
            if analysis:
                print("Async VLM Analysis Result:")
                if analysis.get('choices') and len(analysis['choices']) > 0:
                     message = analysis['choices'][0].get('message')
                     if message and message.get('content'):
                         print(message['content'])
                     else:
                         print("Could not extract content from VLM response.")
                else:
                    print("No choices found in VLM response.")
            else:
                print("Async VLM analysis failed.")
        except Exception as e:
            print(f"Error during async VLM example usage: {e}")
    else:
        print("Skipping async VLM example usage as client could not be initialized.")

if __name__ == '__main__':
    # Run the async main function using asyncio.run()
    asyncio.run(main_test())

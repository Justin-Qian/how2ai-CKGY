from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_api_key():
    """
    Test if the OpenAI API key is properly set and working.
    """
    try:
        # Try to get the API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ Error: OPENAI_API_KEY not found in environment variables")
            return False

        # Initialize the client
        client = OpenAI()

        # Try a simple API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Hello!"}
            ]
        )

        # If we get here, the API key is working
        print("✅ Success: API key is valid and working")
        print(f"Response received: {response.choices[0].message.content}")
        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


if __name__ == "__main__":
    print("Testing OpenAI API key...")
    print(f"Current API key: {os.getenv('OPENAI_API_KEY')}" if os.getenv("OPENAI_API_KEY") else "No API key found")
    test_api_key()

# Text Summarization Evaluation

This project implements text summarization using OpenAI's GPT model and evaluates the summaries using various metrics (BLEU, ROUGE).

## Setup Instructions

### 1. Create a Virtual Environment

```bash
# Create a new virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('wordnet')"
```

### 3. Set up OpenAI API Key

Set your OpenAI API key as an environment variable:

```bash
# On Windows (Command Prompt):
set OPENAI_API_KEY=your_api_key_here

# On Windows (PowerShell):
$env:OPENAI_API_KEY="your_api_key_here"

# On macOS/Linux:
export OPENAI_API_KEY="your_api_key_here"
```

## Project Structure

- `baseline01.py`: Implements text summarization using OpenAI's GPT model
- `metrics.py`: Contains evaluation metrics (BLEU, ROUGE)
- `evaluation.py`: Evaluates generated summaries against reference text
- `processing.py`: Processes input text files

## Usage

1. Process the input text:
```bash
python processing.py
```

2. Generate summary:
```bash
python baseline01.py
```

3. Evaluate the summary:
```bash
python evaluation.py
```

The results will be saved in the `processed_data` directory.

## Requirements

- Python 3.8 or higher
- See `requirements.txt` for package dependencies

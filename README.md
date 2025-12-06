# CS337 Project 2 Part 2: LLM-Based Cooking Assistant

An AI-powered cooking assistant that helps users follow online recipes using Google's Gemini API. The assistant can answer questions about ingredients, steps, timing, substitutions, and equipment while tracking the user's progress through a recipe.

## Features

- **Recipe Scraping**: Automatically extracts recipe content (title, ingredients, steps) from web URLs
- **Interactive Chat**: Natural language conversation with an AI assistant about the recipe
- **Smart Question Answering**: Handles questions about:
  - Ingredient substitutions
  - Step clarifications
  - Cooking times and temperatures
  - Equipment requirements
  - Progress tracking through recipe steps
- **Context-Aware**: Maintains conversation history for follow-up questions
- **Safety-Focused**: Provides food safety guidance and warnings

## Project Structure

```
.
├── assistant.py          # Main application with chat loop
├── scrape.py            # Web scraping functionality for recipes
├── prompt.py            # System prompt for the AI assistant
├── recipe_api.py       # Use the parse function
├── gemini_test.py       # Test script for Gemini API
├── environment.yml      # Conda environment configuration
└── README.md            # This file
```

## Installation

### Prerequisites

- Python 3.11
- Conda (Anaconda or Miniconda)
- Google Gemini API key

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CS337Proj2Part2
   ```

2. **Create the Conda environment**
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment**
   ```bash
   conda activate cs337proj2
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```bash
   touch .env
   ```
   
   Add your Gemini API key to the `.env` file:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```
   
   To get a Gemini API key:
   - Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Sign in with your Google account
   - Create a new API key

## Usage

### Running the Assistant

```bash
python assistant.py
```

### Example Session

```
Paste a recipe URL: https://www.allrecipes.com/recipe/10813/best-chocolate-chip-cookies/
Scraped recipe:

Title: Best Chocolate Chip Cookies
Ingredients:
- 1 cup butter, softened
- 1 cup white sugar
- 1 cup packed brown sugar
...

--- (truncated preview) ---

You: Can I use whole wheat flour instead of all-purpose flour?
Assistant: Yes, you can substitute whole wheat flour, but it will make the cookies denser and give them a nuttier flavor. I recommend using half whole wheat and half all-purpose flour for better texture...

You: What temperature should I bake them at?
Assistant: According to the recipe, bake the cookies at 375°F (190°C)...

You: quit
```

### Testing Gemini API

To test your API connection:

```bash
python gemini_test.py
```

## Dependencies

- **python-dotenv**: Load environment variables from .env file
- **google-genai**: Google Gemini API client
- **requests**: HTTP library for web scraping
- **beautifulsoup4**: HTML parsing library

See `environment.yml` for specific versions.

## How It Works

1. **Scraping**: The `scrape.py` module fetches the recipe webpage and extracts structured information (title, ingredients, steps)

2. **Context Building**: The assistant combines:
   - System prompt (defines behavior and capabilities)
   - Scraped recipe content
   - Conversation history
   - Current user question

3. **LLM Processing**: Sends the combined context to Gemini API for natural language understanding and response generation

4. **Response**: The assistant provides helpful, context-aware answers

## System Prompt

The assistant is guided by a detailed system prompt that defines:
- Capabilities (answering questions, clarifying steps, tracking progress)
- Interaction style (concise, friendly, step-by-step)
- Safety guidelines (food safety, warnings)
- Formatting preferences

See `prompt.py` for the complete system prompt.

## Limitations

- Recipe scraping may not work on all websites due to varying HTML structures
- The assistant relies on the quality of the scraped content
- API rate limits may apply based on your Gemini API tier
- Conversation history grows with each exchange (may hit token limits on very long conversations)

## Environment Management

**Update environment:**
```bash
conda env update -f environment.yml --prune
```

**Remove environment:**
```bash
conda deactivate
conda env remove -n cs337proj2
```

**Export current environment:**
```bash
conda env export > environment.yml
```






# Medical Chatbot - Maternal & Newborn Clinical Guidelines

This project is a specialized Medical Chatbot designed to assist with Maternal and Newborn Clinical Guidelines. It utilizes Retrieval-Augmented Generation (RAG) to provide accurate, context-aware answers based on medical documentation.

![Chatbot Image](img/Screenshot%202025-11-22%20175151.png)

## Features

- **Specialized Knowledge**: Focused specifically on maternal and newborn health protocols.
- **Context-Aware**: Uses retrieved context to answer user queries accurately.
- **Detailed Medical Info**: Provides specific drug names, dosages, and regimens when requested.
- **Multilingual Support**: Capable of answering queries in Sinhala if asked.
- **Web Interface**: Includes a chat interface for easy interaction.

## Project Structure

- [`app.py`](app.py): The main application file (likely Flask or similar) that runs the web server.
- [`store_index.py`](store_index.py): Script to process data and store embeddings in a vector database.
- [`src/prompt.py`](src/prompt.py): Defines the system prompt and rules for the LLM.
- [`src/helper.py`](src/helper.py): Helper functions for data processing or model interaction.
- [`templates/chat.html`](templates/chat.html): HTML template for the chat interface.
- [`static/style.css`](static/style.css): CSS styling for the frontend.
- [`data/`](data/): Directory containing the source medical documents.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add necessary API keys (e.g., OpenAI, Pinecone, etc.).

## Usage

1. **Ingest Data**: Run the indexing script to prepare your vector store.
   ```bash
   python store_index.py
   ```

2. **Run the Application**: Start the chatbot server.
   ```bash
   python app.py
   ```

3. **Access the Chatbot**: Open your browser and navigate to `http://localhost:8080` (or the port specified in `app.py`).

## System Prompt

The bot is governed by specific rules defined in [`src/prompt.// filepath: README.md
# Medical Chatbot - Maternal & Newborn Clinical Guidelines

This project is a specialized Medical Chatbot designed to assist with Maternal and Newborn Clinical Guidelines. It utilizes Retrieval-Augmented Generation (RAG) to provide accurate, context-aware answers based on medical documentation.

## Features

- **Specialized Knowledge**: Focused specifically on maternal and newborn health protocols.
- **Context-Aware**: Uses retrieved context to answer user queries accurately.
- **Detailed Medical Info**: Provides specific drug names, dosages, and regimens when requested.
- **Multilingual Support**: Capable of answering queries in Sinhala if asked.
- **Web Interface**: Includes a chat interface for easy interaction.

## Project Structure

- [`app.py`](app.py): The main application file (likely Flask or similar) that runs the web server.
- [`store_index.py`](store_index.py): Script to process data and store embeddings in a vector database.
- [`src/prompt.py`](src/prompt.py): Defines the system prompt and rules for the LLM.
- [`src/helper.py`](src/helper.py): Helper functions for data processing or model interaction.
- [`templates/chat.html`](templates/chat.html): HTML template for the chat interface.
- [`static/style.css`](static/style.css): CSS styling for the frontend.
- [`data/`](data/): Directory containing the source medical documents.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add necessary API keys (e.g., OpenAI, Pinecone, etc.).

## Usage

1. **Ingest Data**: Run the indexing script to prepare your vector store.
   ```bash
   python store_index.py
   ```

2. **Run the Application**: Start the chatbot server.
   ```bash
   python app.py
   ```

3. **Access the Chatbot**: Open your browser and navigate to `http://localhost:8080` (or the port specified in `app.py`).

## System Prompt

The bot is governed by specific rules defined in [`src/prompt.

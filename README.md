# Vector-Search

This project is a Flask-based web application for vector search and reranking, using Pinecone for vector database operations. It is designed to work with Netflix titles data and embeddings.

## Features
- Upload and embed Netflix titles
- Vector search using Pinecone
- Reranking of search results
- Simple web interface

## Project Structure
- `app.py`: Main Flask application
- `reranking.py`: Reranking logic
- `embed_upload.ipynb`: Jupyter notebook for embedding and uploading data
- `data/`: Contains Netflix CSV data and embeddings
- `static/`: Static files (CSS, favicon)
- `templates/`: HTML templates
- `requirements.txt`: Python dependencies

## Setup
1. Clone the repository and navigate to the project directory.
2. (Recommended) Create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   uv pip install -r requirements.txt
   ```

## Usage
1. Run the Flask app:
   ```
   python app.py
   ```
2. Open your browser and go to `http://127.0.0.1:5000/`.

## Notes
- Make sure you have valid Pinecone and OpenAI API keys set as environment variables if required by your code.
- For more details, see the code and comments in each file.

## License
This project is for educational purposes.

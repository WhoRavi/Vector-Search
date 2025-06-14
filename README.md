# 🎯 Purpose

This app is a vector search tool for Netflix titles. It enables users to search for similar movies or shows using vector embeddings, making content discovery smarter and more intuitive.

---

## 🚀 Features

- 🔍 Search Netflix titles using vector similarity
- 📊 Uses OpenAI for embeddings and Pinecone for vector search
- 🖥️ Simple web interface for easy interaction

---

## 🗂️ Project Structure

- `app.py` — Main Flask application
- `embed_upload.ipynb` — Jupyter notebook for embedding and uploading data
- `requirements.txt` — Python dependencies
- `data/` — Contains Netflix CSV data and embeddings
  - `netflix_titles.csv`
  - `netflix_titles_embedding.csv`
- `static/` — Static files (CSS, favicon)
  - `style.css`
  - `favicon.png`
- `templates/` — HTML templates
  - `index.html`

---

## 🛠️ Installation

1. Clone the repository and navigate to the project directory.
2. (Recommended) Create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

1. Run the Flask app:
   ```
   python app.py
   ```
2. Open your browser and go to `http://127.0.0.1:5000/`.

---

## 📁 Data Files

- `data/netflix_titles.csv` — Raw Netflix titles data
- `data/netflix_titles_embedding.csv` — Embeddings for Netflix titles

---

## 🎨 Static & Templates

- `static/style.css` — App styling
- `static/favicon.png` — App icon
- `templates/index.html` — Main HTML template

---

## 🔑 Notes

- Make sure you have valid Pinecone and OpenAI API keys set as environment variables if required by your code.

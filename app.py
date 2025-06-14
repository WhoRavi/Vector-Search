import os
import pandas as pd
from flask import Flask, render_template, request
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Initialize OpenAI client
client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

# Initialize Pinecone client
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud = cloud, region=region)
pc = Pinecone(api_key = os.getenv('PINECONE_API_KEY'))
 

# dataset = pd.read_excel('netflix_titles_embeddings.xlsx')
dataset = pd.read_csv('./Data/netflix_titles.csv')

index_name = 'netflix-titles'
ns_name = 'movie-tv-shows'

# Create or access Pinecone index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        index_name,
        dimension = 1536,  # dimensionality of text-embedding-3-small
        metric = 'cosine',
        spec = spec
    )

pinecone_index = pc.Index(index_name)

dataset['id'] = dataset.index.astype(str)

# Create a dictionary mapping IDs to a dictionary of all relevant columns
content_mapped = dict(zip(
    dataset.id, 
    dataset[['type', 'title', 'director', 'actors', 'description']].to_dict(orient = 'records')
))

def enhance_query_with_gpt(query):
    sys_query = """
    1. You are an assistant that help users query enhancement for movie title, director, actors, country, release year, rating, duration, description, etc.
    2. Add movie/tv show title, actors, release years, etc if something related is provided in the query. Try to avoid adding description.
    3. Give me in this format:
        a. Type, Title, Director, Actors, Release Year.
        b. Type = Movie/TV Show.
        c. Add upto 3 actors only.
    4. Query can have in Non-English also, translate them to English.
    5. Do not explain extras.
    """
    completion = client.chat.completions.create(
    model = "gpt-4.1-nano",
    messages = [
        {"role": "system", "content": sys_query},
        {"role": "user", "content": f"Enhance the following search query: {query}"}
    ]
    )

    enhanced_query = completion.choices[0].message.content.strip()
    return enhanced_query


def query_article(query, namespace, top_k = 10):

    # Enhance the query using GPT
    enhanced_query = enhance_query_with_gpt(query)

    # query = query.replace("\n", " ")
    query = enhanced_query.replace("\n", " ")
    embedding = client.embeddings.create(input=query, model='text-embedding-3-small').data[0].embedding
    rounded_embedding = [round(value, 15) for value in embedding]

    query_result = pinecone_index.query(vector=rounded_embedding, namespace=namespace, top_k = top_k)

    if not query_result.matches:
        return pd.DataFrame()  # Return an empty DataFrame if no matches

    matches = query_result.matches
    ids = [res.id for res in matches]
    scores = [res.score for res in matches]
    types = [content_mapped[_id]['type'] for _id in ids]
    titles = [content_mapped[_id]['title'] for _id in ids]
    directors = [content_mapped[_id]['director'] for _id in ids]
    actors = [content_mapped[_id]['actors'] for _id in ids]
    descr = [content_mapped[_id]['description'] for _id in ids]

    df = pd.DataFrame({
        'id': ids,
        'Score': scores,
        'Type': types,
        'Title': titles,
        'Director': directors,
        'Actors': actors,
        'Description': descr
    })

    return df

def generate_text_from_results(query, results):
    # Build a context string from the search results.
    # You can decide which fields to include. Here we use all the fields.
    context = ""
    for res in results:
        context += (
            f"Type: {res['Type']}\n"
            f"Title: {res['Title']}\n"
            f"Director: {res['Director']}\n"
            f"Actors: {res['Actors']}\n"
            f"Description: {res['Description']}\n\n"
        ) 
    prompt = (
        f"User Query: {query}\n"
        f"Search Results:\n{context}\n"
        "Based on the above search results, provide a concise, aggregated summary that helps answer the user’s query."
    )
    
    completion = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes search results."},
            {"role": "user", "content": prompt}
        ]
    )
    generated_text = completion.choices[0].message.content.strip()
    return generated_text


app = Flask(__name__)
# Flask App
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        if not query:
            return render_template('index.html', error="Please enter a query")

        result_df = query_article(query, ns_name)  # Name of the namespace
        results = result_df.to_dict(orient='records')

        generated_text = None
        if results:
            generated_text = generate_text_from_results(query, results)
            return render_template('index.html', query=query, results=results, generated_text=generated_text)
        else:
            return render_template('index.html', query=query, error="No results found")
    return render_template('index.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
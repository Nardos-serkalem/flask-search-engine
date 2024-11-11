from flask import Flask, render_template, request
import requests 
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer ###

app = Flask(__name__)

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

def get_wikipedia_summary(query):
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "utf8": 1,
    }
    response = requests.get(WIKIPEDIA_API_URL, params=params)
    search_results = response.json().get("query", {}).get("search", [])
    
    summaries = []
    for result in search_results:
        title = result['title']
        page_id = result['pageid']
        
        summary_params = {
            "action": "query",
            "prop": "extracts|pageimages",
            "exintro": True,
            "explaintext": True,
            "format": "json",
            "pageids": page_id,
            "pithumbsize": 500  
        }
        summary_response = requests.get(WIKIPEDIA_API_URL, params=summary_params)
        page = summary_response.json().get("query", {}).get("pages", {}).get(str(page_id), {})
         #this is image urls 
        image_url = page.get("thumbnail", {}).get("source")
        
        summaries.append({
            "title": title,
            "summary": page.get("extract", ""),
            "url": f"https://en.wikipedia.org/?curid={page_id}",
            "image_url": image_url
        })
        
    return summaries

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')

#Fstep1 etch Wikipedia summaries and images based on the query
    docs = get_wikipedia_summary(query)

#step2  Preprocess and lemmatize summaries
    lemmatizer = WordNetLemmatizer()
    docs_cleaned = [" ".join([lemmatizer.lemmatize(word) for word in doc["summary"].split()]) for doc in docs]

#step3  Vectorize the documents and query
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(docs_cleaned)
    query_vec = vectorizer.transform([query])

#step 4 Calculate cosine similarity
    cosine_similarities = np.dot(X, query_vec.T).toarray().flatten()
    top_results_indices = np.argsort(cosine_similarities)[-5:][::-1]

    results = [
        {
            'document': docs[idx]["summary"],
            'url': docs[idx]["url"],
            'image_url': docs[idx]["image_url"],
            'similarity': cosine_similarities[idx]
        }
        for idx in top_results_indices if cosine_similarities[idx] > 0
    ]

    return render_template("results.html", query=query, results=results)

if __name__ == '__main__':
    app.run(debug=True)

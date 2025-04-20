from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pandas as pd
import re
import nltk
import pickle
import json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

app = Flask(__name__)

# Check if NLTK data is downloaded, if not download it
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize lemmatizer and set of stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Path to model files
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
TFIDF_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
NB_PATH = os.path.join(MODELS_DIR, 'nb_classifier.pkl')
VECTOR_STORE_PATH = os.path.join(MODELS_DIR, 'vector_store.json')

# Banking-related keywords for topic filtering
BANKING_KEYWORDS = [
    'account', 'bank', 'credit', 'debit', 'card', 'loan', 'mortgage', 'interest', 'deposit', 'withdraw',
    'transaction', 'balance', 'statement', 'atm', 'branch', 'check', 'cheque', 'transfer', 'payment',
    'savings', 'checking', 'online banking', 'mobile banking', 'fee', 'charge', 'overdraft', 'pin',
    'password', 'security', 'fraud', 'customer service', 'application', 'apply', 'approve', 'limit',
    'bill', 'pay', 'due date', 'minimum payment', 'reward', 'points', 'cash back', 'credit score',
    'direct deposit', 'routing number', 'account number', 'wire transfer', 'exchange rate', 'currency',
    'investment', 'retirement', 'fund', 'ira', 'cd', 'certificate of deposit', 'money market', 'finwise',
    'passbook', 'bank statement', 'verify', 'identity', 'activation', 'finance', 'financial', 'banking'
]

# Non-banking topics to explicitly filter out
NON_BANKING_TOPICS = [
    'ai', 'artificial intelligence', 'machine learning', 'politics', 'sports', 'entertainment',
    'movies', 'music', 'celebrities', 'cooking', 'recipes', 'travel', 'tourism', 'healthcare',
    'medical', 'diet', 'fitness', 'exercise', 'technology', 'science', 'history', 'geography',
    'language', 'translation', 'gaming', 'video games', 'social media', 'news', 'weather'
]

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def is_banking_related(query):
    """
    Determine if a query is related to banking topics.
    
    Args:
        query (str): The user's query
        
    Returns:
        bool: True if the query is banking-related, False otherwise
        float: Confidence score for the determination
    """
    query_lower = query.lower()
    
    # Check for explicit non-banking topics
    for topic in NON_BANKING_TOPICS:
        if topic in query_lower:
            return False, 0.9  # High confidence that it's not banking-related
    
    # Check for banking keywords
    banking_keyword_count = 0
    for keyword in BANKING_KEYWORDS:
        if keyword in query_lower:
            banking_keyword_count += 1
    
    # Calculate confidence based on keyword matches
    if banking_keyword_count > 0:
        confidence = min(0.5 + (banking_keyword_count * 0.1), 0.95)  # Scale up with more matches, max 0.95
        return True, confidence
    
    # If no keywords matched, use additional heuristics or default to not banking-related
    # Some very generic questions might be banking-related but don't contain specific keywords
    generic_banking_queries = [
        'help', 'how does this work', 'what can you do', 'what services', 
        'tell me about', 'who are you', 'what is finwise'
    ]
    
    for generic_query in generic_banking_queries:
        if generic_query in query_lower:
            return True, 0.6  # Medium confidence for generic but potentially relevant queries
    
    # Default case - if we're unsure, we'll be conservative and say it's not banking-related
    return False, 0.7

def load_models():
    try:
        with open(TFIDF_PATH, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)

        with open(NB_PATH, 'rb') as f:
            nb_classifier = pickle.load(f)

        with open(VECTOR_STORE_PATH, 'r') as f:
            vector_store = json.load(f)

        return tfidf_vectorizer, nb_classifier, vector_store
    except FileNotFoundError as e:
        app.logger.error(f"Error loading models: {e}")
        return None, None, None

def classify_intent(query, tfidf_vectorizer, nb_classifier):
    # Preprocess the query
    processed_query = preprocess_text(query)

    # Transform using TF-IDF
    query_tfidf = tfidf_vectorizer.transform([processed_query])

    # Predict the category
    category = nb_classifier.predict(query_tfidf)[0]

    # Get prediction probabilities
    probs = nb_classifier.predict_proba(query_tfidf)[0]
    max_prob = max(probs)

    return category, float(max_prob)

def retrieve_relevant_responses(query, category, vector_store, top_k=5):
    # Load the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode the query
    query_embedding = model.encode(query)

    # Convert stored embeddings back to numpy arrays
    stored_embeddings = np.array(vector_store['embeddings'])

    # Get indices of responses from the specified category
    category_indices = [i for i, cat in enumerate(vector_store['categories']) if cat == category]

    if not category_indices:
        # If no responses in the category, use all responses
        category_indices = list(range(len(vector_store['responses'])))

    # Filter embeddings and responses by category
    category_embeddings = np.array([stored_embeddings[i] for i in category_indices])
    category_responses = [vector_store['responses'][i] for i in category_indices]

    # Calculate cosine similarity
    similarities = cosine_similarity([query_embedding], category_embeddings)[0]

    # Get indices of top k similar responses
    top_indices = similarities.argsort()[-top_k:][::-1]

    # Get the top k responses and their similarity scores
    top_responses = [category_responses[i] for i in top_indices]
    top_scores = [float(similarities[i]) for i in top_indices]

    return top_responses, top_scores

def generate_llm_response(query, relevant_responses, groq_api_key, is_banking_topic=True):
    # Initialize Groq client
    client = Groq(api_key=groq_api_key)
    
    # If the query is not banking-related, return a polite refusal
    if not is_banking_topic:
        return "I'm FinWise, a banking assistant, and I can only help with questions related to banking, finance, accounts, loans, credit cards, and other banking services. If you have questions about these topics, I'd be happy to assist you."

    # Construct the prompt for the LLM
    prompt = f"""You are FinWise, a helpful and friendly banking assistant. You ONLY answer questions related to banking, finance, accounts, loans, credit cards, and other banking services.
    A user has asked the following question: "{query}"

    Here are some relevant pieces of information that might help you answer:

    {' '.join([f'- {resp}' for resp in relevant_responses])}

    IMPORTANT RULES:
    1. ONLY respond to banking and finance related questions.
    2. If the question is about ANY other topic (like AI, technology, science, entertainment, etc.), politely state that you can only help with banking-related inquiries.
    3. Never make up information about banking services that isn't supported by your knowledge.
    4. Keep answers concise, helpful and focused on the specific banking query.
    5. If you're not sure about the answer, suggest contacting customer service for more information.

    The response should be conversational, concise, and focused on answering their specific banking question.
    """

    # Generate response from Groq's Llama model
    try:
        completion = client.chat.completions.create(
            model="llama3-8b-8192",  # or your preferred Llama model
            messages=[
                {"role": "system", "content": "You are FinWise, a banking assistant ONLY. You EXCLUSIVELY provide information about banking, financial services, and related topics. For ANY other topics, you politely decline to answer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )
        return completion.choices[0].message.content
    except Exception as e:
        app.logger.error(f"Error generating LLM response: {e}")
        return "I apologize, but I'm having trouble generating a response at the moment. Please try again later or contact customer service for assistance."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def process_query():
    data = request.json
    user_query = data.get('query', '')
    groq_api_key = os.environ.get('GROQ_API_KEY', '')
    
    if not groq_api_key:
        return jsonify({
            "error": "GROQ API key not set. Please set the GROQ_API_KEY environment variable."
        }), 500
    
    # First check if query is banking-related
    is_banking_topic, topic_confidence = is_banking_related(user_query)
    
    # If not banking-related, return a standard response without calling the models
    if not is_banking_topic:
        return jsonify({
            "query": user_query,
            "detected_category": "non_banking",
            "confidence": topic_confidence,
            "response": "I'm FinWise, a banking assistant, and I can only help with questions related to banking, finance, accounts, loans, credit cards, and other banking services. If you have questions about these topics, I'd be happy to assist you.",
            "relevant_sources": [],
            "source_relevance_scores": []
        })
    
    # Load models and vector store
    tfidf_vectorizer, nb_classifier, vector_store = load_models()
    
    if not all([tfidf_vectorizer, nb_classifier, vector_store]):
        return jsonify({
            "error": "Models not found. Please train the models first."
        }), 500

    # Classify the intent
    category, confidence = classify_intent(user_query, tfidf_vectorizer, nb_classifier)

    # Retrieve relevant responses
    relevant_responses, similarity_scores = retrieve_relevant_responses(
        user_query, category, vector_store, top_k=5
    )

    # Generate LLM response
    response = generate_llm_response(user_query, relevant_responses, groq_api_key, is_banking_topic)

    # Return the result with metadata
    return jsonify({
        "query": user_query,
        "detected_category": category,
        "confidence": confidence,
        "response": response,
        "relevant_sources": relevant_responses,
        "source_relevance_scores": similarity_scores
    })

if __name__ == '__main__':
    app.run(debug=True)
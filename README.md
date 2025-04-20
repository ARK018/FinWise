# FinWise - Smart Banking Assistant

FinWise is an intelligent banking assistant powered by machine learning and large language models. It uses intent classification and semantic search to provide accurate, contextual responses to banking queries.

## Features

- 🧠 **ML-Based Intent Classification**: Automatically detects the category of banking queries
- 🔍 **Vector Store Knowledge Base**: Retrieves relevant information based on user queries
- 💬 **LLM-Powered Responses**: Generates natural, conversational answers using Groq's Llama3 model
- 🔒 **Banking-Focused**: Strictly answers banking-related questions only
- 📱 **Responsive UI**: Clean, modern interface built with HTML, Tailwind CSS, and JavaScript

## Project Structure

```
FinWise/
├── app.py                # Main Flask application
├── train_models.py       # Script to train and save models
├── models/               # Directory storing ML models
│   ├── tfidf_vectorizer.pkl
│   ├── nb_classifier.pkl
│   └── vector_store.json
├── static/               # Static assets
│   ├── css/
│   │   └── styles.css    # Custom styling
│   └── js/
│       └── script.js     # Frontend JavaScript
└── templates/            # HTML templates
    └── index.html        # Main application page
```

## How It Works

1. **User Input**: User enters a banking-related question
2. **Topic Detection**: System verifies the query is banking-related
3. **Intent Classification**: ML model categorizes the query (credit card, loan, account, etc.)
4. **Contextual Lookup**: Vector store retrieves the most relevant information
5. **Response Generation**: LLM generates a helpful, accurate response based on the retrieved context
6. **Response Delivery**: The answer is displayed to the user with metadata about the process

## Installation

### Prerequisites

- Python 3.8+
- pip
- Virtual environment (recommended)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/ARK018/FinWise.git
cd finwise
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install flask nltk scikit-learn sentence-transformers groq pandas
```

4. Set your Groq API key:

```bash
# On Windows
set GROQ_API_KEY=your_groq_api_key_here
# On macOS/Linux
export GROQ_API_KEY=your_groq_api_key_here
```

## Usage

1. Train the models (only needed once):

```bash
python train_models.py
```

2. Run the application:

```bash
python app.py
```

3. Open your browser and go to `http://127.0.0.1:5000`

## Banking Categories

FinWise can answer questions about:

- Credit cards
- Loans and mortgages
- Bank accounts
- ATM services
- Passbooks and statements
- Online and mobile banking

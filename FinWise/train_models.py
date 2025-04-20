import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import json

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and set of stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Sample banking FAQ data with categories
banking_data = [
    # Credit Card queries
    {"query": "How do I apply for a credit card?", "response": "You can apply for a credit card through our mobile app, website, or by visiting any branch. You'll need to provide identification, proof of income, and complete an application form.", "category": "credit_card"},
    {"query": "What is the credit limit on my card?", "response": "Credit limits are determined based on your income, credit history, and other financial factors. You can check your current limit through online banking, our mobile app, or by calling customer service.", "category": "credit_card"},
    {"query": "How do I increase my credit limit?", "response": "You can request a credit limit increase through online banking or our mobile app. We'll review your account history, payment record, and current financial situation before making a decision.", "category": "credit_card"},
    {"query": "I've lost my credit card. What should I do?", "response": "Immediately report your lost card by calling our 24/7 hotline or blocking it through our mobile app. We'll deactivate the card and issue a replacement, which should arrive within 3-5 business days.", "category": "credit_card"},
    {"query": "How do I activate my new credit card?", "response": "You can activate your new card through our mobile app, online banking, by calling the activation number on the card sticker, or by making a purchase with PIN.", "category": "credit_card"},
    {"query": "What are the rewards on my credit card?", "response": "Our credit cards offer various rewards including cashback, travel points, or merchant discounts depending on your card type. You can view your specific rewards program details in our mobile app or online banking portal.", "category": "credit_card"},
    {"query": "When is my credit card bill due?", "response": "Your credit card payment due date is typically 21 days after your statement date. You can find the exact date on your monthly statement, in online banking, or our mobile app.", "category": "credit_card"},
    {"query": "How do I dispute a charge on my credit card?", "response": "To dispute a transaction, log into online banking or our mobile app and select the transaction you want to contest. Alternatively, call our customer service. Please note disputes must be raised within 60 days of the transaction.", "category": "credit_card"},

    # Loan queries
    {"query": "How do I apply for a personal loan?", "response": "You can apply for a personal loan through our mobile app, website, or by visiting any branch. The application requires identification documents, income proof, and details about your financial situation.", "category": "loan"},
    {"query": "What documents do I need for a home loan?", "response": "For a home loan, you'll need ID proof, address proof, income documents (salary slips or tax returns), property documents, and details of existing loans. Additional documents may be required based on your specific situation.", "category": "loan"},
    {"query": "What is the interest rate on your loans?", "response": "Our current interest rates range from 7.5% to 12.5% for personal loans, 6.5% to 9% for home loans, and 8% to 14% for auto loans, depending on your credit profile, loan amount, and tenure.", "category": "loan"},
    {"query": "How long does loan approval take?", "response": "Personal loan approval typically takes 1-3 business days, auto loans 1-2 days, and home loans 5-7 business days after all required documents are submitted. Pre-approved customers may receive instant approval.", "category": "loan"},
    {"query": "Can I pay off my loan early?", "response": "Yes, you can make partial or full prepayments on your loan. Personal and auto loans may have a prepayment penalty of 2-3% if paid within 12 months. Home loans typically allow prepayment without penalty after 6 months.", "category": "loan"},
    {"query": "How do I check my loan balance?", "response": "You can check your current loan balance through online banking, our mobile app, by visiting any branch, or by calling our customer service hotline.", "category": "loan"},
    {"query": "What happens if I miss a loan payment?", "response": "Missing a payment may result in late fees (typically 2-3% of the amount due), can negatively impact your credit score, and may result in higher interest rates on future loans. Please contact us immediately if you anticipate payment difficulties.", "category": "loan"},
    {"query": "Can I restructure my loan?", "response": "Yes, loan restructuring is possible in case of financial hardship. Options include extending the tenure, temporary reduction in EMI, or interest rate adjustments. Please contact our loan servicing department to discuss your specific situation.", "category": "loan"},

    # Account queries
    {"query": "How do I open a new bank account?", "response": "You can open an account online through our website, via our mobile app, or by visiting any branch with your ID proof, address proof, and a passport-sized photograph. The process takes approximately 15-20 minutes.", "category": "account"},
    {"query": "What are the different types of accounts?", "response": "We offer several account types including Savings Accounts, Current Accounts, Salary Accounts, Fixed Deposit Accounts, and Recurring Deposit Accounts. Each has different features, minimum balance requirements, and benefits.", "category": "account"},
    {"query": "What is the minimum balance for savings account?", "response": "The minimum balance requirement for our standard savings account is $500 in urban branches and $250 in rural branches. Premium accounts have higher requirements. Zero-balance basic accounts are available for qualifying customers.", "category": "account"},
    {"query": "How do I check my account balance?", "response": "You can check your balance through online banking, our mobile app, at any ATM, by visiting a branch, through SMS banking by texting BAL to our service number, or by calling our interactive voice response system.", "category": "account"},
    {"query": "I've lost my debit card. What should I do?", "response": "Immediately block your card through our mobile app or online banking. Alternatively, call our 24/7 helpline to report the loss. Once blocked, you can request a replacement card which will be delivered within 7-10 business days.", "category": "account"},
    {"query": "How do I update my address?", "response": "You can update your address through online banking, our mobile app, or by visiting any branch with a valid address proof. Changes are typically processed within 2 business days.", "category": "account"},
    {"query": "How do I generate account statements?", "response": "Account statements can be generated for any period through online banking or our mobile app. You can download them as PDF documents or request physical statements by visiting a branch or calling customer service.", "category": "account"},
    {"query": "What are the charges for money transfers?", "response": "NEFT transfers are free of charge. RTGS transfers are free for amounts over $10,000. IMPS transfers have a nominal fee of $0.50 for amounts up to $5,000 and $1 for higher amounts. International wire transfers have variable fees based on amount and destination.", "category": "account"},

    # Passbook/Statement queries
    {"query": "How do I get my passbook updated?", "response": "You can update your passbook at any branch by inserting it into the passbook kiosk or by handing it to a teller. All branches also have self-service passbook printing machines that you can use during banking hours.", "category": "passbook"},
    {"query": "Can I get a duplicate passbook?", "response": "Yes, you can request a duplicate passbook by visiting your home branch and submitting a request form. There is a nominal fee of $5 for a new passbook, and it will be issued on the same day.", "category": "passbook"},
    {"query": "How far back can I see transactions in my passbook?", "response": "A standard passbook can show transactions for about 1-2 years depending on your activity level. For older transactions, you would need to request account statements from the bank.", "category": "passbook"},
    {"query": "How do I get a statement for tax purposes?", "response": "Tax statements are available through online banking under the 'Statements & Documents' section. You can generate interest certificates, loan statements, and other tax-related documents. These can also be requested at any branch.", "category": "passbook"},
    {"query": "How often should I update my passbook?", "response": "We recommend updating your passbook at least once a month to keep track of all transactions. However, with online banking and mobile apps, many customers now prefer electronic statements over passbook updates.", "category": "passbook"},
    {"query": "Can I get my account statement emailed to me?", "response": "Yes, you can subscribe to e-statements through online banking or our mobile app. Monthly statements will be automatically emailed to your registered email address. This service is free of charge and environmentally friendly.", "category": "passbook"},
    {"query": "How do I read my bank statement?", "response": "Your bank statement lists all transactions chronologically with deposits shown as credits and withdrawals as debits. Each entry includes the date, description, amount, and running balance. Call customer service if you need help understanding specific entries.", "category": "passbook"},
    {"query": "How long are my statements stored?", "response": "We retain electronic statements for 7 years and they are all accessible through online banking. Physical statements may need to be specially requested for periods older than 12 months.", "category": "passbook"},

    # ATM queries
    {"query": "Where is the nearest ATM?", "response": "You can locate the nearest ATM through our mobile app using the 'ATM/Branch Locator' feature which uses GPS to show ATMs near your current location. You can also enter a specific address to find ATMs in that area.", "category": "atm"},
    {"query": "What is the daily withdrawal limit at ATMs?", "response": "The standard daily ATM withdrawal limit is $1,000 for regular debit cards and $2,000 for premium cards. For temporary limit increases, you can request through online banking or by calling customer service.", "category": "atm"},
    {"query": "The ATM didn't dispense cash but debited my account", "response": "Please contact our 24/7 customer service immediately to report this issue. After verification, the debited amount will be reversed to your account within 7 working days as per banking regulations.", "category": "atm"},
    {"query": "How do I change my ATM PIN?", "response": "You can change your ATM PIN through our mobile app, online banking, by visiting any of our ATMs and selecting the 'PIN Change' option, or by calling customer service to request a new PIN which will be sent by mail.", "category": "atm"},
    {"query": "Are there charges for using other banks' ATMs?", "response": "You can use other bank ATMs free of charge for the first 3 transactions (5 for premium accounts) per month. Additional transactions incur a fee of $1.50 per cash withdrawal and $0.50 per balance inquiry.", "category": "atm"},
    {"query": "Do you have cardless cash withdrawal?", "response": "Yes, we offer cardless cash withdrawal through our mobile app. You can generate a temporary code that can be used at our ATMs to withdraw cash without your physical card. The code is valid for 30 minutes.", "category": "atm"},
    {"query": "My card is stuck in the ATM. What should I do?", "response": "Please contact our emergency helpline immediately. If the ATM belongs to our bank, we'll arrange for the branch to retrieve your card when they open. If it's another bank's ATM, we'll block the card and issue a replacement.", "category": "atm"},
    {"query": "Can I deposit cash at ATMs?", "response": "Yes, you can deposit cash at our Advanced ATMs which have cash deposit functionality. These machines accept loose notes and provide instant credit to your account with a transaction receipt.", "category": "atm"},

    # Online/Mobile Banking queries
    {"query": "How do I register for online banking?", "response": "You can register for online banking through our website by clicking 'First Time User' and following the instructions. You'll need your account number, debit card details, and registered mobile number to complete the process.", "category": "online_banking"},
    {"query": "I forgot my online banking password", "response": "You can reset your password on our login page by clicking 'Forgot Password' and following the security verification steps. A temporary password will be sent to your registered mobile number or email address.", "category": "online_banking"},
    {"query": "How do I download the mobile banking app?", "response": "Our mobile banking app is available on both Apple App Store and Google Play Store. Search for 'FinWise Banking' and download the app. After installation, you can login with your existing online banking credentials.", "category": "online_banking"},
    {"query": "Is mobile banking secure?", "response": "Yes, our mobile banking uses multiple security layers including encryption, two-factor authentication, biometric verification, and transaction monitoring systems. We also employ device binding and automatic session timeouts for additional security.", "category": "online_banking"},
    {"query": "How do I set up bill payments online?", "response": "To set up bill payments, log into online banking or our mobile app, go to 'Bill Payments', select 'Add Biller', choose the category, enter the biller details, and save. You can then schedule one-time or recurring payments.", "category": "online_banking"},
    {"query": "Can I transfer money to other banks online?", "response": "Yes, you can transfer funds to other banks through NEFT, RTGS, or IMPS services via our online banking or mobile app. Simply add the beneficiary with their account details and initiate the transfer.", "category": "online_banking"},
    {"query": "What are the transaction limits for online banking?", "response": "Standard online banking daily limits are $5,000 for third-party transfers and $10,000 for own account transfers. Premium accounts have higher limits of $25,000 and $50,000 respectively. You can request temporary limit increases through customer service.", "category": "online_banking"},
    {"query": "The app is showing an error. What should I do?", "response": "Try these troubleshooting steps: refresh the app, check your internet connection, clear the app cache, ensure you're using the latest version, or restart your device. If the problem persists, contact our technical support team with the error message.", "category": "online_banking"}
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

def main():
    # Convert the data to a DataFrame
    df = pd.DataFrame(banking_data)
    
    # Apply preprocessing to queries
    df['processed_query'] = df['query'].apply(preprocess_text)
    
    # Create features and labels for training
    X = df['processed_query']
    y = df['category']
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Create TF-IDF vectors
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Train Naive Bayes classifier
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_tfidf, y_train)
    
    # Evaluate the classifier
    y_pred = nb_classifier.predict(X_test_tfidf)
    print("Naive Bayes Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create embeddings using Sentence Transformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df['response_embedding'] = df['response'].apply(lambda x: model.encode(x))
    
    # Create model directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the TF-IDF vectorizer
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    
    # Save the Naive Bayes classifier
    with open('models/nb_classifier.pkl', 'wb') as f:
        pickle.dump(nb_classifier, f)
    
    # Create and save the vector store
    embeddings_list = [emb.tolist() for emb in df['response_embedding'].values]
    
    vector_store = {
        'responses': df['response'].tolist(),
        'categories': df['category'].tolist(),
        'embeddings': embeddings_list
    }
    
    with open('models/vector_store.json', 'w') as f:
        json.dump(vector_store, f)
    
    print("Training complete. Models and vector store saved in the 'models' directory.")

if __name__ == "__main__":
    main()
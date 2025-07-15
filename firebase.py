import os
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import find_dotenv, load_dotenv


# Load environment variables from .env file
dotenv_path = find_dotenv() # If .env file is found, load it
if dotenv_path:
    load_dotenv(dotenv_path)

FIREBASE_CRED = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")

def initialize_firebase():
    # Initialize Firebase Admin SDK
    cred = credentials.Certificate("llm-automated-research-firebase-adminsdk.json")
    firebase_admin.initialize_app(cred)

    # Initialize Firestore
    db = firestore.client()
    return db 
db = initialize_firebase()
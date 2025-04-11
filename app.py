import imaplib
import email
from email.header import decode_header
import pandas as pd
import logging
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify
import time
import threading

app = Flask(__name__)

# Replace with your Gmail and App Password
EMAIL = "D3CODE1105@gmail.com"
PASSWORD = "icfm pnrr shle jzvl"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_text(text):
    """Clean and normalize the email text."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove excessive spaces
    return text

def load_training_data():
    """Load and clean the training dataset."""
    try:
        df = pd.read_csv("email_training_data.csv")
        df.columns = df.columns.str.strip()
        if 'body' not in df.columns or 'queue' not in df.columns:
            raise ValueError("CSV must contain 'body' and 'queue' columns.")
        df["body"] = df["body"].fillna("No content provided").apply(clean_text)
        logging.info(f"Loaded {len(df)} training samples.")
        return df
    except Exception as e:
        logging.error(f"Failed to load training data: {e}")
        raise

def train_classifier():
    """Train the classifier using Random Forest."""
    try:
        df = load_training_data()

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            lowercase=True,
            max_features=5000
        )
        X = vectorizer.fit_transform(df["body"])
        y = df["queue"]

        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X, y)

        logging.info("Random Forest classifier trained.")
        return vectorizer, classifier
    except Exception as e:
        logging.error(f"Classifier training failed: {e}")
        raise

# Global model & vectorizer
vectorizer, classifier = train_classifier()

def fetch_emails():
    """Fetch the latest emails from Gmail."""
    try:
        imap = imaplib.IMAP4_SSL("imap.gmail.com")
        imap.login(EMAIL, PASSWORD)
        imap.select("inbox")
        status, messages = imap.search(None, "ALL")
        if status != "OK":
            logging.warning("No emails found.")
            return []

        email_ids = messages[0].split()
        emails = []

        for email_id in email_ids:  # Get all emails
            status, msg_data = imap.fetch(email_id, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject, encoding = decode_header(msg["Subject"])[0]
                    subject = subject.decode(encoding or "utf-8") if isinstance(subject, bytes) else subject
                    from_ = msg.get("From")
                    date_ = msg.get("Date")
                    body = ""

                    if msg.is_multipart():
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            if content_type == "text/plain" and "attachment" not in str(part.get("Content-Disposition")):
                                body = part.get_payload(decode=True).decode(errors="ignore")
                                break
                    else:
                        body = msg.get_payload(decode=True).decode(errors="ignore")

                    if not body:
                        continue

                    emails.append({
                        "subject": subject,
                        "from": from_,
                        "date": date_,
                        "body": body
                    })

        imap.logout()
        logging.info(f"Fetched {len(emails)} emails.")
        return emails
    except Exception as e:
        logging.error(f"Failed to fetch emails: {e}")
        return []

def classify_and_save_emails(emails):
    """Classify and save emails based on queue category."""
    try:
        categorized = {}

        for email_data in emails:
            body_cleaned = clean_text(email_data["body"])
            X_email = vectorizer.transform([body_cleaned])
            predicted = classifier.predict(X_email)[0]

            email_data["predicted_queue"] = predicted
            logging.info(f"Email Subject: '{email_data.get('subject')}' predicted as: {predicted}")

            if predicted not in categorized:
                categorized[predicted] = []
            categorized[predicted].append(email_data)

        for queue, email_list in categorized.items():
            df = pd.DataFrame(email_list)
            filename = f"{queue.lower().replace(' ', '_')}_emails.csv"
            try:
                existing = pd.read_csv(filename)
                df = pd.concat([existing, df], ignore_index=True)
            except FileNotFoundError:
                pass
            df.to_csv(filename, index=False)
            logging.info(f"Saved {len(email_list)} emails to '{filename}'.")

    except Exception as e:
        logging.error(f"Failed to classify and save emails: {e}")
        raise

def email_polling():
    """Continuously poll for new emails."""
    while True:
        logging.info("Checking for new emails...")
        emails = fetch_emails()
        if emails:
            classify_and_save_emails(emails)
        time.sleep(30)  # Poll every 30 seconds

# Flask endpoints
@app.route("/scrape-emails", methods=["GET"])
def scrape_emails():
    try:
        emails = fetch_emails()
        if not emails:
            return jsonify({"error": "No emails to process"}), 400
        classify_and_save_emails(emails)
        return jsonify({"message": "Emails processed and saved"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/classify-email", methods=["POST"])
def classify_email():
    try:
        email_body = request.json.get("body")
        if not email_body:
            return jsonify({"error": "Email body is required"}), 400
        clean_body = clean_text(email_body)
        X_email = vectorizer.transform([clean_body])
        prediction = classifier.predict(X_email)
        return jsonify({"predicted_queue": prediction[0]}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/evaluate", methods=["GET"])
def evaluate():
    try:
        df = load_training_data()
        X = vectorizer.transform(df["body"])
        y = df["queue"]
        y_pred = classifier.predict(X)
        accuracy = accuracy_score(y, y_pred)
        return jsonify({"accuracy": accuracy}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Start email polling in a background thread
    threading.Thread(target=email_polling, daemon=True).start()
    app.run(debug=True)

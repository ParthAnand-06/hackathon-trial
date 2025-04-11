import imaplib
import email
from email.header import decode_header
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import logging

app = Flask(__name__)

EMAIL = "D3code1105@gmail.com"  # Update with your email
PASSWORD = "icfm pnrr shle jzvl"  # Update with your app password

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_training_data():
    """Load the training data for the classifier."""
    try:
        train_df = pd.read_csv("email_training_data.csv")
        train_df.columns = train_df.columns.str.strip()  # Remove extra spaces from column names
        
        if 'body' not in train_df.columns or 'type' not in train_df.columns:
            raise ValueError("The CSV must contain 'body' and 'type' columns.")

        train_df["body"] = train_df["body"].fillna("No content provided")
        
        logging.info(f"Loaded {len(train_df)} training samples.")
        return train_df
    except Exception as e:
        logging.error(f"Error loading training data: {e}")
        raise

def train_classifier():
    """Train the email classification model."""
    try:
        train_df = load_training_data()

        vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Using bi-grams
        X_train = vectorizer.fit_transform(train_df["body"])
        y_train = train_df["type"]

        classifier = KNeighborsClassifier(n_neighbors=3)
        classifier.fit(X_train, y_train)

        logging.info("Classifier trained successfully.")
        return vectorizer, classifier
    except Exception as e:
        logging.error(f"Error training classifier: {e}")
        raise

vectorizer, classifier = train_classifier()

def fetch_emails():
    """Fetch the latest emails from Gmail using IMAP."""
    try:
        # Connect to Gmail
        imap = imaplib.IMAP4_SSL("imap.gmail.com")
        imap.login(EMAIL, PASSWORD)
        logging.info("Logged into Gmail successfully.")
        
        # Select the inbox
        imap.select("inbox")
        
        # Search for all emails
        status, messages = imap.search(None, "ALL")
        if status != "OK" or not messages[0]:
            logging.error("No emails found or search failed.")
            return []
        
        # Get email IDs
        email_ids = messages[0].split()
        logging.info(f"Found {len(email_ids)} email IDs.")
        
        emails = []
        for email_id in email_ids[-5:]:  # Fetching last 5 emails for demo purposes
            status, msg_data = imap.fetch(email_id, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject, encoding = decode_header(msg["Subject"])[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding if encoding else "utf-8")
                    from_ = msg.get("From")
                    date_ = msg.get("Date")
                    body = ""
                    
                    # Extract the body of the email
                    if msg.is_multipart():
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            content_disposition = str(part.get("Content-Disposition"))
                            if "attachment" not in content_disposition:
                                if content_type == "text/plain":
                                    body = part.get_payload(decode=True).decode()
                                    break
                    else:
                        body = msg.get_payload(decode=True).decode()

                    if not body:
                        logging.warning(f"Email from {from_} has no body.")
                        continue

                    emails.append({
                        "subject": subject,
                        "from": from_,
                        "date": date_,
                        "body": body,
                    })
        
        # Log out of the IMAP session
        imap.logout()
        logging.info(f"Fetched {len(emails)} emails.")
        return emails
    except Exception as e:
        logging.error(f"Error fetching emails: {e}")
        return []

def classify_and_save_emails(emails):
    """Classify and save emails to CSV files based on predicted categories."""
    try:
        classified_emails = []

        # Classify each email and store it in classified_emails
        for email_data in emails:
            body = email_data["body"]

            # Transform the body of the email using the trained vectorizer
            email_vectorized = vectorizer.transform([body])
            predicted_category = classifier.predict(email_vectorized)
            email_data["predicted_category"] = predicted_category[0]
            classified_emails.append(email_data)

        # Convert list to DataFrame
        all_emails_df = pd.DataFrame(classified_emails)
        all_emails_df.to_csv("classified_emails.csv", index=False)
        logging.info("All emails classified and saved to 'classified_emails.csv'.")

        # Save to individual category files
        target_categories = [
            "Technical Support",
            "Product Support",
            "Customer Service",
            "IT Support",
            "Billing and Payments"
        ]

        # Save emails to category-specific CSV files
        for category in target_categories:
            category_df = all_emails_df[all_emails_df["predicted_category"] == category]
            if not category_df.empty:
                filename = category.lower().replace(" ", "_") + "_emails.csv"
                category_df.to_csv(filename, index=False)
                logging.info(f"Saved {len(category_df)} emails to '{filename}'.")
            else:
                logging.info(f"No emails found for category '{category}'.")

    except Exception as e:
        logging.error(f"Error classifying and saving emails: {e}")
        raise

@app.route("/scrape-emails", methods=["GET"])
def scrape_emails():
    """Fetch, classify, and save emails to CSV files."""
    try:
        emails = fetch_emails()

        if not emails:
            return jsonify({"error": "No emails found to classify."}), 400

        classify_and_save_emails(emails)

        return jsonify({"message": "Emails classified and saved to CSV files."}), 200

    except Exception as e:
        logging.error(f"Error during email scraping: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/classify-email", methods=["POST"])
def classify_email():
    """Classify a single email body."""
    try:
        email_body = request.json.get("body")
        if not email_body:
            return jsonify({"error": "Email body is required"}), 400

        # Transform the email body and classify
        email_vectorized = vectorizer.transform([email_body])
        predicted_label = classifier.predict(email_vectorized)
        return jsonify({"predicted_category": predicted_label[0]}), 200

    except Exception as e:
        logging.error(f"Error during email classification: {e}")
        return jsonify({"error": str(e)}), 500
from sklearn.metrics import accuracy_score

def evaluate_model(vectorizer, classifier):
    """Evaluate model accuracy using the training dataset."""
    try:
        # Load the training data again
        train_df = load_training_data()

        # Vectorize the training email bodies
        X_train = vectorizer.transform(train_df["body"])

        # True labels
        y_true = train_df["type"]

        # Predicted labels
        y_pred = classifier.predict(X_train)

        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)

        logging.info(f"Training Accuracy: {accuracy:.4f}")
        print(f"Training Accuracy: {accuracy:.4f}")

        return accuracy
    except Exception as e:
        logging.error(f"Error evaluating model accuracy: {e}")
        return None
@app.route("/evaluate", methods=["GET"])
def evaluate():
    try:
        acc = evaluate_model(vectorizer, classifier)
        if acc is not None:
            return jsonify({"training_accuracy": acc}), 200
        else:
            return jsonify({"error": "Could not evaluate model."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

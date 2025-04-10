import imaplib
import email
from email.header import decode_header
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import logging
app = Flask(__name__)
EMAIL = "D3code1105@gmail.com"  
PASSWORD = "icfm pnrr shle jzvl"  

def load_training_data():
    try:
       
        train_df = pd.read_csv("email_training_data.csv")
        train_df.columns = train_df.columns.str.strip()  

        if 'body' not in train_df.columns or 'type' not in train_df.columns:
            raise ValueError("The CSV must contain 'body' and 'type' columns.")

        train_df["body"] = train_df["body"].fillna("No content provided")

        logging.info(f"Loaded {len(train_df)} training samples.")
        return train_df
    except Exception as e:
        logging.error(f"Error loading training data: {e}")
        raise

def train_classifier():
    try:
        train_df = load_training_data()

        vectorizer = TfidfVectorizer(ngram_range=(1, 2))  
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
    try:
        imap = imaplib.IMAP4_SSL("imap.gmail.com")
        imap.login(EMAIL, PASSWORD)

        imap.select("inbox")

        status, messages = imap.search(None, "ALL")

        # Get email IDs
        email_ids = messages[0].split()
        emails = []

        for email_id in email_ids[-5:]:
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

                    emails.append({
                        "subject": subject,
                        "from": from_,
                        "date": date_,
                        "body": body,
                    })

        imap.logout()
        return emails

    except Exception as e:
        logging.error(f"Error fetching emails: {e}")
        return []

def classify_and_save_emails(emails):
    try:
        classified_emails = []

        for email_data in emails:
            body = email_data["body"]

            email_vectorized = vectorizer.transform([body])
            predicted_category = classifier.predict(email_vectorized)
            email_data["predicted_category"] = predicted_category[0]
            classified_emails.append(email_data)


        all_emails_df = pd.DataFrame(classified_emails)
        all_emails_df.to_csv("classified_emails.csv", index=False)

        logging.info("All emails classified and saved to 'classified_emails.csv'.")


        for category in all_emails_df["predicted_category"].unique():
            category_df = all_emails_df[all_emails_df["predicted_category"] == category]
            category_df.to_csv(f"{category}_emails.csv", index=False)
            logging.info(f"Saved {category} emails to '{category}_emails.csv'.")

    except Exception as e:
        logging.error(f"Error classifying and saving emails: {e}")
        raise


@app.route("/scrape-emails", methods=["GET"])
def scrape_emails():
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
    try:

        email_body = request.json.get("body")
        if not email_body:
            return jsonify({"error": "Email body is required"}), 400


        email_vectorized = vectorizer.transform([email_body])
        predicted_label = classifier.predict(email_vectorized)
        return jsonify({"predicted_category": predicted_label[0]}), 200

    except Exception as e:
        logging.error(f"Error during email classification: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
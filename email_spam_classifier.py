# Simple Email Spam Classifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training data
emails = [
    "Congratulations! You have won a free iPhone.",
    "Claim your lottery prize now.",
    "Get cheap medicines at low prices.",
    "Let's meet for project discussion tomorrow.",
    "Please review the attached project file.",
    "You have been selected for a cash reward."
]

# Labels: 1 = Spam, 0 = Not Spam
labels = [1, 1, 1, 0, 0, 1]

# Convert text into numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emails)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X, labels)

print("✅ Model trained successfully!\n")

# Loop for user input
while True:
    email = input("Enter an email message (or type 'exit' to quit): ")
    if email.lower() == "exit":
        print("Exiting program...")
        break

    X_test = vectorizer.transform([email])
    prediction = model.predict(X_test)[0]

    if prediction == 1:
        print("🚫 This email is SPAM!\n")
    else:
        print("✅ This email is NOT SPAM!\n")
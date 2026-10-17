import re
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLP preprocessing using NLTK
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    words = re.findall(r"\b[a-z0-9]+\b", text)
    words = [stemmer.stem(word) for word in words]
    return " ".join(words)

# FAQ dataset: questions and answers
faqs = [
    {
        "question": "What courses do you offer?",
        "answer": "We offer various undergraduate and postgraduate courses."
    },
    {
        "question": "What are the college timings?",
        "answer": "Please check your college timetable for the current timings."
    },
    {
        "question": "How can I pay my fees?",
        "answer": "Contact the college accounts office for fee payment details."
    },
    {
        "question": "How do I apply for admission?",
        "answer": "Contact the admissions office for application details."
    },
    {
        "question": "Is hostel accommodation available?",
        "answer": "Please contact the college hostel office for availability."
    },
    {
        "question": "How can I contact the college?",
        "answer": "Visit the official college website for contact information."
    },
    {
        "question": "Are placements available?",
        "answer": "Contact the placement cell for placement and training details."
    }
]

# Prepare FAQ questions for similarity matching
questions = [faq["question"] for faq in faqs]
processed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(processed_questions)

def get_answer(user_question):
    processed_input = preprocess(user_question)
    input_vector = vectorizer.transform([processed_input])

    similarities = cosine_similarity(input_vector, faq_vectors)[0]
    best_index = similarities.argmax()
    best_score = similarities[best_index]

    if best_score < 0.15:
        return "Sorry, I couldn't find a suitable answer. Please ask another question."

    return faqs[best_index]["answer"]

# Chatbot interface in the terminal
print("=" * 40)
print("       COLLEGE FAQ CHATBOT")
print("=" * 40)
print("Ask a question. Type 'exit' to stop.")

while True:
    user_input = input("\nYou: ").strip()

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Bot: Thank you! Have a nice day.")
        break

    if not user_input:
        print("Bot: Please enter a question.")
        continue

    answer = get_answer(user_input)
    print("Bot:", answer)
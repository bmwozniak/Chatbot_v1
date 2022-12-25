# Program chatbota, który uczy się na podstawie interakcji z użytkownikiem i jest wytrenowany
# na wybranej bazie danych w formacie epub.
# Chatbot ten korzysta z biblioteki NLTK do przetwarzania
# tekstu i biblioteki scikit-learn do tworzenia modelu uczenia przyrostowego.
# Chatbot również korzysta z biblioteki ebooklib do konwersji plików epub na pliki tekstowe.
#
# zawiera  funkcje, takie jak:
#
#     Funkcja do dodawania nowych pytań i odpowiedzi do bazy danych chatbota
#     Funkcja do zapisywania bazy danych chatbota do pliku
#     Funkcja do wczytywania bazy danych chatbota z pliku
#     Obsługa wielu różnych baz danych w formacie epub




import ebooklib
import nltk
import random
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

# Funkcja do konwersji plików epub na pliki tekstowe
def convert_epub_to_text(filepath):
    book = ebooklib.epub.read_epub(filepath)
    text = ""
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            text += item.get_content().decode("utf-8")
    return text

# Funkcja do tokenizacji tekstu
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

# Funkcja do przygotowania tekstu do uczenia
def prepare_text_for_learning(text):
    text = text.lower()
    stopwords = nltk.corpus.stopwords.words("english")
    tokens = tokenize(text)
    tokens = [token for token in tokens if token not in stopwords]
    return " ".join(tokens)


# Funkcja do dodawania nowych pytań i odpowiedzi do bazy danych chatbota
def add_to_database(questions, answers):
    # Dopasuj nowe pytania do wektorizera
    vectorizer.fit(questions)
    # Dodaj nowe pytania i odpowiedzi do modelu uczenia przyrostowego
    classifier.partial_fit(vectorizer.transform(questions), answers)


# Funkcja do zapisywania bazy danych chatbota do pliku
def save_database(filename):
    # Otwórz plik do zapisu
    with open(filename, "wb") as file:
        # Zapisz wektorizer i model do pliku
        pickle.dump((vectorizer, classifier), file)


# Funkcja do wczytywania bazy danych chatbota z pliku
def load_database(filename):
    # Otwórz plik do odczytu
    with open(filename, "rb") as file:
        # Wczytaj wektorizer i model z pliku
        global vectorizer, classifier
        vectorizer, classifier = pickle.load(file)


# Funkcja do odpowiedzi chatbota na pytanie użytkownika
def generate_answer(question):
    # Przygotuj pytanie do uczenia
    prepared_question = prepare_text_for_learning(question)
    # Dopasuj pytanie do wektorizera
    question_vector = vectorizer.transform([prepared_question])
    # Otrzymaj odpowiedź od modelu
    answer = classifier.predict(question_vector)[0]
    # Zwróć odpowiedź
    return answer


# Pętla główna chatbota
while True:
    # Pobierz pytanie od użytkownika
    question = input("Enter your question: ")
    # Jeśli pytanie to "exit", zakończ pętlę
    if question.lower() == "exit":
        break
    # Jeśli pytanie to "add", dodaj nowe pytania i odpowiedzi do bazy danych
    elif question.lower() == "add":
        print("Enter new questions and answers, one per line. Type 'done' when finished.")


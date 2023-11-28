import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
import pandas as pd

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')


# Function for preprocessing text
def preprocess_text(text):
    sentences = sent_tokenize(text)
    preprocessed_sentences = []
    stop_words = set(stopwords.words('english'))

    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
        preprocessed_sentences.append(words)

    return preprocessed_sentences


# Function to train Word2Vec model and save details to CSV
def train_word2vec_model(sentences, window_size, embedding_size, model_name):
    model = Word2Vec(sentences, window=window_size, vector_size=embedding_size, sg=0)  # sg=0 for CBOW, sg=1 for Skip-gram

    # Save model details to CSV
    details_df = pd.DataFrame({
        'Model Name': [model_name],
        'Window Size': [window_size],
        'Embedding Size': [embedding_size],
        'Vocabulary Size': [len(model.wv)],
        'Most Similar Words': [model.wv.most_similar('example_word', topn=5)],
        # Replace 'example_word' with a word from your data
    })

    details_df.to_csv(f'{model_name}-details.csv', index=False, mode='w')

    # Save model to file if needed
    # model.save(f'{model_name}.model')

    return details_df


# Main script
book_files = ["hamlet.txt", "macbeth.txt", "othello.txt", "romeoJuliet.txt", "tempest.txt"]

analysis_df = pd.DataFrame()

for book_file in book_files:
    with open(book_file, 'r', encoding='utf-8') as file:
        book_text = file.read()

    # Preprocess text
    preprocessed_sentences = preprocess_text(book_text)

    # Train Word2Vec models with different parameters
    window_sizes = [3, 5]
    embedding_sizes = [100, 200]

    for window_size in window_sizes:
        for embedding_size in embedding_sizes:
            model_name = f'model_W{window_size}_E{embedding_size}'
            details_df = train_word2vec_model(preprocessed_sentences, window_size, embedding_size, model_name)
            analysis_df = analysis_df.append(details_df, ignore_index=True)

# Save analysis details to CSV
analysis_df.to_csv('analysis.csv', index=False, mode='a')
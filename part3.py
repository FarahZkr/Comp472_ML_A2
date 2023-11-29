import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
import pandas as pd
import random

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

file_path = 'A2-DataSet/synonym.csv' #loading the data set
dataset = pd.read_csv(file_path)
dataset_length = len(dataset)


def write_to_file_1(name, message):
    with open(f"{name}-details.csv", 'a') as file:
        file.write(message)


def write_to_file_2(message):
    with open("analysis.csv", 'a') as file:
        file.write(message)

# Function for preprocessing text - takes in the array of files - tokenize each sentences into words
def preprocess_text(text):
    sentences = sent_tokenize(text)
    preprocessed_sentences = []
    stop_words = set(stopwords.words('english'))

    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
        preprocessed_sentences.append(words)

    return preprocessed_sentences

def find_closest_synonym(model, question_word, synonym_words):
    # try:


        # if question_word in model.wv.index_to_key:
            # Check if any of the potential synonym words are in the model's vocabulary
            # if any(word in model.wv.index_to_key
            #        for word in [question_word] + synonym_words):
            #     similarities = [(word, model.wv.similarity(question_word, word)) for word in synonym_words]
            #     # syn with largest similarity
            #     closest_synonym = max(similarities, key=lambda x: x[1])[0]
            #     # if question word in model
            #     return closest_synonym

        if question_word in model.wv:
            similarity= []
            print (question_word, synonym_words)
            for choice in synonym_words:
                # print (choice)
                if choice in model.wv:
                    print ("**")
                    similarity.append(model.wv.similarity(question_word,choice))
                else:
                    return  "Synonym not found"
                index_closest_synonym = similarity.index(max(similarity))
            closest_synonym = synonym_words[index_closest_synonym]
            print (closest_synonym)
        else:
            print ("??")
            return "Synonyms not found."
        return closest_synonym
    #     else:
    #         return "Synonyms not found."
    # except KeyError:
    #     return "Synonyms not found."

# Main script
book_files = ["hamlet.txt", "macbeth.txt", "othello.txt", "romeoJuliet.txt", "tempest.txt", "captain.txt", "willowWeaver.txt"]

# dataframe
analysis_df = pd.DataFrame()

book_text = ""
for book_file in book_files:
    with open(book_file, 'r', encoding='utf-8') as file:
        book_text += file.read()

# Preprocess text
preprocessed_sentences = preprocess_text(book_text)

# Train Word2Vec models with different parameters
window_sizes = [5, 10]
embedding_sizes = [25, 50]

models = []
model_names =[]

for window_size in window_sizes:
    for embedding_size in embedding_sizes:
        model_name = f'model_W{window_size}_E{embedding_size}'
        model_names.append(model_name)
        model = Word2Vec(preprocessed_sentences, window=window_size, vector_size=embedding_size)
        model.train(preprocessed_sentences, total_examples=len(preprocessed_sentences), epochs=10)
        # print(model.wv['hardly'])
        models.append(model)


for i, model in enumerate(models):
    correct_labels = 0
    model_name = model_names[i]
    tmp = dataset_length
    vocab_size = len(model.wv.key_to_index)
    print(model_name + ": " + str(vocab_size))
    print(f"Model {i}: {model_name}")
    for index, row in dataset.iterrows():
        question = row['question']
        answer = row['answer']
        synonyms = [row[f'{i}'] for i in range(0, 4)]

        to_write = f"{question},{answer}"

        similar_word = find_closest_synonym(model, question, synonyms)
        if similar_word != "Synonyms not found.":
            to_write += f",{similar_word}"
            if similar_word != answer:
                to_write += ",wrong"
            else:
                to_write += ",correct"
                correct_labels += 1
        else:
            to_write += f",{random.choice(synonyms).lower()},guess"
            tmp -= 1

        to_write += "\n"
        write_to_file_1(model_name, to_write)
        to_write = ""
    if tmp == 0:
        total = 0
    else:
        total = correct_labels / tmp
    analysis_file = f"{model_name},{vocab_size},{correct_labels},{tmp},{total}\n"
    write_to_file_2(analysis_file)
# Save analysis details to CSV
analysis_df.to_csv('analysis.csv', index=False, header=False, mode='a')
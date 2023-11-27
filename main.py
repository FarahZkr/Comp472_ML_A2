import pandas as pd
import gensim.downloader
import random

# ======================================
word2vec_model_name = 'word2vec-google-news-300'
word2vec_model = gensim.downloader.load(word2vec_model_name)

file_path = 'A2-DataSet/synonym.csv'
dataset = pd.read_csv(file_path)
dataset_length = len(dataset)


def write_to_file_1(name, message):
    with open(f"{name}-details.csv", 'a') as file:
        file.write(message)


def write_to_file_2(message):
    with open("analysis.csv", 'a') as file:
        file.write(message)


def find_closest_synonym(model, question_word, synonym_words):
    try:
        if question_word in model.index_to_key:
            if any(word in model.index_to_key for word in [question_word] + synonym_words):
                similarities = [(word, model.similarity(question_word, word)) for word in synonym_words]
                closest_synonym = max(similarities, key=lambda x: x[1])[0]
                return closest_synonym
            else:
                return "Synonyms not found."
        else:
            return "Synonyms not found."
    except KeyError:
        return "Synonyms not found."


# ======================================

# Models of same size: Twitter Corpus & Wiki GigaWord Corpus: size = 100
# Fast test model: fasttext-wiki-news-subwords-300
# Wiki-GigaWord: glove-wiki-gigaword-300

fasttest_model_name = 'fasttext-wiki-news-subwords-300'
fasttest_model = gensim.downloader.load(fasttest_model_name)

giga_model_name = 'glove-wiki-gigaword-300'
giga_model = gensim.downloader.load(giga_model_name)

# Same models with different sizes:
# Twitter: glove-twitter-25
# Twitter: glove-twitter-100

twitter_model_name = 'glove-twitter-25'
twitter_model = gensim.downloader.load(twitter_model_name)

twitter_model_name2 = 'glove-twitter-100'
twitter_model2 = gensim.downloader.load(twitter_model_name2)

models = [word2vec_model, fasttest_model, giga_model, twitter_model, twitter_model2]
model_names = [word2vec_model_name, fasttest_model_name, giga_model_name, twitter_model_name, twitter_model_name2]

for i, model in enumerate(models):
    correct_labels = 0
    model_name = model_names[i]
    tmp = dataset_length
    vocab_size = len(model.index_to_key)
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

# ======================================

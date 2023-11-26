import pandas as pd
import gensim.downloader
import random

# ======================================
# region Part-1

model_name = 'word2vec-google-news-300'
word2vec_model = gensim.downloader.load(model_name)

file_path = 'A2-DataSet/synonym.csv'
dataset = pd.read_csv(file_path)
dataset_length = len(dataset)
vocab_size = len(word2vec_model.index_to_key)


def write_to_file_1(message):
    with open(f"{model_name}-details.csv", 'a') as file:
        file.write(message)


def write_to_file_2(message):
    with open("analysis.csv", 'w') as file:
        file.write(message)


def find_closest_synonym(questionWord, synonym_words):
    try:
        if questionWord in word2vec_model.index_to_key:
            if any(word in word2vec_model.index_to_key for word in [questionWord] + synonym_words):
                similarities = [(word, word2vec_model.similarity(questionWord, word)) for word in synonym_words]
                closest_synonym = max(similarities, key=lambda x: x[1])[0]
                return closest_synonym
            else:
                closest_word = word2vec_model.most_similar(questionWord, topn=1)[0][0].lower()
                return closest_word
        else:
            return "Synonyms not found."
    except KeyError:
        return "Synonyms not found."


correct_labels = 0
for index, row in dataset.iterrows():
    question = row['question']
    answer = row['answer']
    synonyms = [row[f'{i}'] for i in range(0, 4)]

    to_write = f"{question},{answer}"

    similar_word = find_closest_synonym(question, synonyms)
    if similar_word != "Synonyms not found." and similar_word in synonyms:
        to_write += f",{similar_word}"
        if similar_word != answer:
            to_write += ",wrong"
        else:
            to_write += ",correct"
            correct_labels += 1
    elif similar_word != "Synonyms not found.":
        to_write += f",{similar_word},guess"
        dataset_length -= 1
    else:
        to_write += f",{random.choice(word2vec_model.index_to_key).lower()},guess"
        dataset_length -= 1
    to_write += "\n"
    write_to_file_1(to_write)
    to_write = ""


total = correct_labels / dataset_length
analysis_file = f"{model_name},{vocab_size},{correct_labels},{dataset_length},{total}"
write_to_file_2(analysis_file)

# endregion
# ======================================
# region Part-2

# Models of same size: Twitter Corpus & Wiki GigaWord Corpus: size = 100
# Twitter model: glove-twitter-100
# Wiki-GigaWord: glove-wiki-gigaword-100

# twitter_model_name = 'glove-twitter-100'
# twitter_model = gensim.downloader.load(twitter_model_name)
#
# giga_model_name = 'glove-wiki-gigaword-100'
# giga_model = gensim.downloader.load(giga_model_name)
#
# # Same models with different sizes:
# # Twitter: glove-twitter-50
# # Wiki-GigaWord: glove-wiki-gigaword-300
#
# twitter_model_name2 = 'glove-twitter-50'
# twitter_model2 = gensim.downloader.load(twitter_model_name2)
#
# giga_model_name2 = 'glove-wiki-gigaword-300'
# giga_model2 = gensim.downloader.load(giga_model_name2)


# endregion
# ======================================
# region Part-3

# endregion

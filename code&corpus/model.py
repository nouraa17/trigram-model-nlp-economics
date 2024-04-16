# install python-docx
import docx
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import re

#preparing the corpus -> all lowercase and remove special characters
def preprocess_text(text):
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if re.match('^[a-zA-Z]+$', word)]
    return filtered_words
#############################################################################################################################
#extracting the text from the word file
def read_word_file(filename):
    doc = docx.Document(filename)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return ' '.join(full_text)
#############################################################################################################################
#building the trigram model using ngram
def build_trigram_model_from_word_file(filename):
    text = read_word_file(filename)
    trigram_model = {}
    words = preprocess_text(text)
    for trigram in ngrams(words, 3, pad_right=True, pad_left=True):
        context = (trigram[0], trigram[1])
        word = trigram[2]
        if None not in context:
            if context not in trigram_model:
                trigram_model[context] = {}
            if word not in trigram_model[context]:
                trigram_model[context][word] = 0
            trigram_model[context][word] += 1

    for context, next_words in trigram_model.items():
        total_occurrences = sum(next_words.values())
        for word in next_words:
            trigram_model[context][word] /= total_occurrences

    return trigram_model
#############################################################################################################################
filename = r'economic_corpus.docx'
trigram_model = build_trigram_model_from_word_file(filename)
#############################################################################################################################
#saving the model to text file
def save_trigram_model_to_text(trigram_model, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for context, next_words in trigram_model.items():
            f.write(f"{context[0]} {context[1]}:\n")
            for word, probability in next_words.items():
                f.write(f"  {word}: {probability}\n")
            f.write('\n')
save_trigram_model_to_text(trigram_model, 'trigram_model.txt')
#############################################################################################################################
#read the text file & convert to javascript object to use it in js file
def read_trigram_data(file_path):
    trigram_data = {}
    with open(file_path, 'r') as file:
        current_bigram = None
        for line in file:
            line = line.strip()
            if line:
                if line.endswith(':'):
                    current_bigram = line[:-1]
                    trigram_data[current_bigram] = []
                else:
                    word, probability = line.split(': ')
                    trigram_data[current_bigram].append([word, float(probability)])
    return trigram_data

def convert_to_nested_object(trigram_data):
    trigram_model = {}
    for bigram, word_prob_pairs in trigram_data.items():
        trigram_model[bigram] = [[pair[0], float(pair[1])] for pair in word_prob_pairs]
    return trigram_model
#############################################################################################################################
trigram_data = read_trigram_data('trigram_model.txt')
trigram_model = convert_to_nested_object(trigram_data)
#############################################################################################################################
#saving the object to a new text file
def save_trigram_model(trigram_model, file_path):
    with open(file_path, 'w') as file:
        file.write("{\n")
        for bigram, words in trigram_model.items():
            file.write(f'  "{bigram}": {words},\n')
        file.write("};")

save_trigram_model(trigram_model, 'trigram_data.txt')





from nltk.corpus import stopwords
from nltk import FreqDist, word_tokenize, sent_tokenize
from string import punctuation


def text_summarize(text, num_lines=3):

    # Remove the stopwords and Punctuation
    stop_words = stopwords.words('english') + list(punctuation) + ['``']

    # Create the Bag of words
    words = [word for word in word_tokenize(text.lower()) if word not in stop_words]

    # Make a list of Sentences with StopWords
    original_sentences = []
    for sentence in sent_tokenize(text):
        original_sentences.append(sentence)

    # Make a list of sentences without StopWords
    sentences = []
    for sentence in sent_tokenize(text.lower()):
        op_sent = []
        for word in word_tokenize(sentence):
            if word not in stop_words:
                op_sent.append(word)
        sent = ' '.join(op_sent)
        sentences.append(sent)

    # Find the frequency Distribution of the BOW and convert to dict
    freq_words = dict(FreqDist(words))

    # Find the weight of each sentence and make a dictionary
    # whose key gives the weight and value is the corresponding sentence
    sent_dict = dict()
    for sentence in original_sentences:
        sent_weight = 0
        sent_words = word_tokenize(sentence)
        for sent_word in sent_words:
            if sent_word.lower() not in stop_words:
                weight = freq_words[sent_word.lower()]
                sent_weight += weight
        sent_dict[sent_weight] = sentence

    # Sort the weights in the form of a list in descending order
    sorted_weight_list = list(sent_dict.keys())
    sorted_weight_list = sorted(sorted_weight_list, key=int, reverse=True)

    # Create a new index dict whose keys contain the enumeration and the values are the sorted weights
    index_dict = dict()
    for index, weight in enumerate(sorted_weight_list):
        index_dict[index] = weight

    # Print the most weighted sentences corresponding to the number of lines
    final_output = []
    for i in range(num_lines):
        final_output.append(sent_dict[index_dict[i]])
    final_output = '\n'.join(final_output)

    return final_output

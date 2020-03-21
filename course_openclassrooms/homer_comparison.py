####################################################################
# Compare the frequency of the words from the Iliad and the Odyssey
# to analyse the evolution of the thematics.
####################################################################

import os

from pyspark import SparkContext
from nltk.corpus import stopwords

sc = SparkContext()
english_stop_words = set(stopwords.words("english"))

def load_text(text_path):
    vocabulary = sc.textFile(text_path) \
                    .flatMap(lambda line: line.lower().split()) \
                    .map(lambda word: word.strip(".,;:?!\"-' ")) \
                    .filter(lambda word: word is not None and len(word) > 0) \
                    .filter(lambda word: word not in english_stop_words)

    # vocabulary.persist()

    word_count = vocabulary.count()

    word_freq = vocabulary.map(lambda word: (word, 1)) \
                            .reduceByKey(lambda count1, count2: count1 + count2) \
                            .map(lambda key_value: (key_value[0], key_value[1] / float(word_count)))

    return word_freq

# Load text vocabulary
iliad = load_text('Spark/data/iliad.mb.txt')
odyssey = load_text('Spark/data/odyssey.mb.txt')

# Compare frequencies
joined_words = iliad.fullOuterJoin(odyssey) \
                    .map(lambda word_freqs: (word_freqs[0], (word_freqs[1][1] or 0) - (word_freqs[1][0] or 0)))

# Select the 10 most appearing and disappearing words
emerging_words = joined_words.takeOrdered(10, lambda word_freqDiff: -word_freqDiff[1])
disappearing_words = joined_words.takeOrdered(10, lambda word_freqDiff: word_freqDiff[1])

# Print the result
for word, freq_diff in emerging_words:
    print("%.2f" % (freq_diff*10000), word)
for word, freq_diff in disappearing_words[::-1]:
    print("%.2f" % (freq_diff*10000), word)

# input("press ctrl+c to exit")  # never use in production, only use locally for debugging
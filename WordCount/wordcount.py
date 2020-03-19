from datetime import datetime
from os import path
import sys

from pyspark import SparkContext

if sys.argv < 2:
    print("Please pass file path as arguments.")
    sys.exit()

file_path = sys.argv[1]

if not path.exists(file_path):
    print("File {} not found.".format(file_path))
    sys.exit()

# Add text to SparkContext
sc = SparkContext()
lines = sc.textFile(file_path)

# Select and print the 100 most frequent words
word_count = lines.flatMap(lambda line: line.split(" ")) \
                    .map(lambda x: (x, 1)) \
                    .reduceByKey(lambda count1, count2: count1 + count2) \
                    .takeOrdered(100, lambda i: -i[1])

for (word, count) in word_count:
    print(word.encode('utf8'), count)

# List all words in the file
word_count = lines.flatMap(lambda line: line.split(" ")) \
                    .distinct() \
                    .takeOrdered(100)

for (word) in word_count:
    print(word.encode('utf8'))

# Test persistance of the RDD
def countWordsWithoutPersistance(file_path):
    now = datetime.now()
    word_count = lines.flatMap(lambda line: line.split(" ")) \
                        .distinct()
                        
    nb_words = word_count.count()
    word_count.takeOrdered(nb_words)
    print("Without persistance: {}".format(datetime.now() - now))


def countWordsWithPersistance(file_path):
    now = datetime.now()
    word_count = lines.flatMap(lambda line: line.split(" ")) \
                        .distinct()
    word_count.persist()

    nb_words = word_count.count()
    word_count.takeOrdered(nb_words)
    print("With persistance: {}".format(datetime.now() - now))


countWordsWithoutPersistance(file_path)
countWordsWithPersistance(file_path)
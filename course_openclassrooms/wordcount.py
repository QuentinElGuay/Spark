from datetime import datetime
import os
import sys

from pyspark import SparkContext

## Initialization
# Create a big file for the exercise by writing 100 times the Iliad in the same file
file_path = '/tmp/iliad100.txt'
with open('Spark/data/iliad.mb.txt', 'r') as iliad:
    file_object = open(file_path, 'a')
    text = iliad.read()

    for i in range(100):
        file_object.write(text)
    file_object.close()

# Add text to SparkContext
sc = SparkContext()
lines = sc.textFile(file_path)

# Select and print the 100 most frequent words
print('Those are the 100 most frequent words in the Iliad with their number of occurences:')
word_count = lines.flatMap(lambda line: line.split(" ")) \
                    .map(lambda word: word.strip(".,;:?!\"-'")) \
                    .map(lambda word: (word, 1)) \
                    .reduceByKey(lambda count1, count2: count1 + count2) \
                    .takeOrdered(100, lambda i: -i[1])

for (word, count) in word_count:
    print(word.encode('utf8'), count / 100)  # we joined 100 times the text.

# List all words in the file using persistance of the RDD
word_count = lines.flatMap(lambda line: line.split(" ")) \
                    .map(lambda word: word.strip(".,;:?!\"-'")) \
                    .distinct()
word_count.persist()

nb_words = word_count.count()
print("The Iliad contains {} distinct words. Here is a sample:".format(nb_words))

# Print a sample of 1% of the words
words = word_count.sample(False, 0.01).collect()
for word in words:
    print(word.encode('utf8'))

## Remove the file
os.remove(file_path)
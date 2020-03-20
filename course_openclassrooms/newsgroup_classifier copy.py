####################################################################
# Create a classifier to predict the topics of newsgroup discussions
# Use sparkML library
####################################################################

from pyspark.sql import Row, SparkSession
from pyspark import SparkContext
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import CountVectorizer, StringIndexer

sc = SparkContext()
spark = SparkSession.builder.master("local") \
                    .appName("Sparkml_course") \
                    .getOrCreate()

def load_dataframe(path):
    rdd = sc.textFile(path)\
        .map(lambda line: line.split())\
        .map(lambda words: Row(label=words[0], words=words[1:]))
    return spark.createDataFrame(rdd)


# load dataframes
train_data = load_dataframe("Spark/data/20ng-train-all-terms.txt")
test_data = load_dataframe("Spark/data/20ng-test-all-terms.txt")

# Count word frequency
vectorizer = CountVectorizer(inputCol="words", outputCol="bag_of_words")
vectorizer_transformer = vectorizer.fit(train_data)
train_bag_of_words = vectorizer_transformer.transform(train_data)
test_bag_of_words = vectorizer_transformer.transform(test_data)

# Create numeric labels
label_indexer = StringIndexer(inputCol="label", outputCol="label_index")
label_indexer_transformer = label_indexer.fit(train_bag_of_words)

train_bag_of_words = label_indexer_transformer.transform(train_bag_of_words)
test_bag_of_words = label_indexer_transformer.transform(test_bag_of_words)

# Create classifier
classifier = NaiveBayes(
    labelCol="label_index",
    featuresCol="bag_of_words",
    predictionCol="label_index_predicted"
)

# Train model
classifier_transformer = classifier.fit(train_bag_of_words)
test_predicted = classifier_transformer.transform(test_bag_of_words)

test_predicted.select("label_index", "label_index_predicted").limit(10).show()

# Evaluate model
evaluator = MulticlassClassificationEvaluator(
    labelCol="label_index",
    predictionCol="label_index_predicted",
    metricName='accuracy'
)
accuracy = evaluator.evaluate(test_predicted)

# Should return 83% of accuracy in prediction
print("Accuracy = {:.2f}".format(accuracy))
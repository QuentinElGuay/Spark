#############################################################################
# Classifier to try predict if a sentence is from the Iliad or the Odissey
# Use pyspark.ml.pipelines
############################################################################

from pyspark.sql import Row, SparkSession
from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF, StopWordsRemover

sc = SparkContext()
spark = SparkSession.builder.master("local") \
                    .appName("Sparkml_course") \
                    .getOrCreate()

def load_dataframe(path, label):   # Could probably use a tokenizer here
    rdd = sc.textFile(path)\
        .map(lambda line: line.split())\
        .map(lambda words: [word.strip(".,;:?!\"-' ") for word in words])\
        .filter(lambda x: len(x) > 0)\
        .map(lambda words: Row(label=label, words=words))
    return rdd


# load books and create DataFrame
iliad_rdd = load_dataframe("Spark/data/iliad.mb.txt", 0)
odyssey_rdd = load_dataframe("Spark/data/odyssey.mb.txt", 1)
df = spark.createDataFrame(iliad_rdd.union(odyssey_rdd))
train_data, test_data = df.randomSplit([0.75, 0.25], seed=1)

# 
StopWordsRemover.loadDefaultStopWords("english")

# Create all steps of the pipeline
stopWordsRemover = StopWordsRemover(inputCol="words", outputCol="filteredWords")
hashingTF = HashingTF(inputCol="filteredWords", outputCol="rawFeatures")
idf = IDF(inputCol="rawFeatures", outputCol="features")
classifier = NaiveBayes(
    labelCol="label",
    featuresCol="features",
    predictionCol="label_predicted"
)

# Create the pipeline
pipeline = Pipeline(stages=[stopWordsRemover, hashingTF, idf, classifier])
pipeline_model = pipeline.fit(train_data)
test_predicted = pipeline_model.transform(test_data)

test_predicted.select("filteredWords", "label", "label_predicted").limit(40).show()

# Evaluate model
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="label_predicted",
    metricName='accuracy'
)
accuracy = evaluator.evaluate(test_predicted)
# Should return 83% which is better than 80% with the CountVectorizer
print("Accuracy = {:.2f}".format(accuracy))
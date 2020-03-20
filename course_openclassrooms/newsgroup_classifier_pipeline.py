####################################################################
# Same as newsgroup_classifier.py
# Use pyspark.ml.pipelines
# User tf-idf instead of simple counter
####################################################################

from pyspark.sql import Row, SparkSession
from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF, StringIndexer

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

# Create all steps of the pipeline
# Use TF-IDF instead of the countVectorizer to get more precise results
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
idf = IDF(inputCol="rawFeatures", outputCol="features")
label_indexer = StringIndexer(inputCol="label", outputCol="label_index")
classifier = NaiveBayes(
    labelCol="label_index",
    featuresCol="features",
    predictionCol="label_index_predicted"
)

# Create the pipeline
pipeline = Pipeline(stages=[hashingTF, idf, label_indexer, classifier])
pipeline_model = pipeline.fit(train_data)
test_predicted = pipeline_model.transform(test_data)

test_predicted.select("label_index", "label_index_predicted").limit(10).show()

# Evaluate model
evaluator = MulticlassClassificationEvaluator(
    labelCol="label_index",
    predictionCol="label_index_predicted",
    metricName='accuracy'
)
accuracy = evaluator.evaluate(test_predicted)
# Should return 83% which is better than 80% with the CountVectorizer
print("Accuracy = {:.2f}".format(accuracy))
# Spark
Experimentations with spark

## Install Spark
/!\ Spark only works on Linux / MacOs.
1. [Download Spark](http://spark.apache.org/downloads.html). I used Spark 2.4.5.
2. Unzip Spark on your computer.
3. You'll need Java installed. Carefull Spark 2.4.5 runs on Java 8. If you have an upper version of Java installed, you can follow [this thread](https://stackoverflow.com/questions/53583199/pyspark-error-unsupported-class-file-major-version-55) to solve your problem.
4. Go to your Spark folder and run `./spark-2.4.5-bin-hadoop2.7/bin/spark-submit --h` to make sure Spark is installed.
5. Install `pyspark`:
``` bash
pip install pyspark
```
6. Run scripts and have fun:
``` bash
./spark-2.4.5-bin-hadoop2.7/bin/spark-submit --master local[4] ./Spark/WordCount/wordcount.py
```

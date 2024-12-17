from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, trim
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import from_json, col, lower, regexp_replace, when
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, Word2Vec
from pyspark.ml import Pipeline
import logging
import time
from pyspark.ml import PipelineModel
import os, shutil
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from base_model import train_model



MODEL_PATH = os.path.join(os.getcwd(), "sarcasm_model")


def process_data():
 
    spark = SparkSession.builder \
        .appName("KafkaSparkStreaming") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.2") \
        .getOrCreate()

    
    df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "my_topic") \
        .load()

    

    schema = StructType([StructField("class", StringType()), StructField("text", StringType())])

    df = df.selectExpr("CAST(value AS STRING)") \
        .select(from_json(col("value"), schema).alias("data")) \
        .select("data.*")


    df = df.withColumn("text", lower(col("text")))

    model = PipelineModel.load(MODEL_PATH)
    predictions = model.transform(df)
    

    query = predictions.writeStream \
        .outputMode("append") \
        .format("console") \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    process_data()







from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, trim
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import from_json, col, lower, regexp_replace, when
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, Word2Vec
from pyspark.ml import Pipeline
import logging
import time
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator



def process_data():
 
    spark = SparkSession.builder \
        .appName("KafkaSparkStreaming") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()


    def hyperparameter_optimization(training_data):
        tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", pattern="\\W")
        remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
        word2vec = Word2Vec(vectorSize=100, minCount=0, inputCol="filtered_tokens", outputCol="features")


        models = {
            "Logistic Regression": (
                LogisticRegression(featuresCol="features", labelCol="class_num"),
                ParamGridBuilder()
                .addGrid(word2vec.vectorSize, [50, 100, 200])
                .addGrid(word2vec.minCount, [1, 5])
                .addGrid(LogisticRegression.regParam, [0.01, 0.1, 1.0])
                .build()
            ),
            "Random Forest": (
                RandomForestClassifier(featuresCol="features", labelCol="class_num"),
                ParamGridBuilder()
                .addGrid(word2vec.vectorSize, [50, 100, 200])
                .addGrid(word2vec.minCount, [1, 5])
                .addGrid(RandomForestClassifier.numTrees, [10, 50])
                .addGrid(RandomForestClassifier.maxDepth, [5, 10])
                .build()
            ),
            "Gradient-Boosted Trees": (
                GBTClassifier(featuresCol="features", labelCol="class_num"),
                ParamGridBuilder()
                .addGrid(word2vec.vectorSize, [50, 100, 200])
                .addGrid(word2vec.minCount, [1, 5])
                .addGrid(GBTClassifier.maxDepth, [5, 10])
                .addGrid(GBTClassifier.maxIter, [10, 50])
                .build()
            )
        }

        best_model = None
        best_f1_score = 0.0
        best_model_name = ""
        best_params = None

        evaluator = BinaryClassificationEvaluator(metricName="f1", labelCol="class_num")


        for model_name, (model, param_grid) in models.items():
            pipe = Pipeline(stages=[tokenizer, remover, word2vec, model])

            crossval = CrossValidator(
                estimator=pipe,
                estimatorParamMaps=param_grid,
                evaluator=evaluator,
                numFolds=3  
            )


            cv_model = crossval.fit(training_data)
            f1_score = evaluator.evaluate(cv_model.transform(training_data))

            print(f"Model: {model_name}, F1 Score: {f1_score}")

  
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_model = cv_model.bestModel
                best_model_name = model_name
                best_params = cv_model.getEstimatorParamMaps()

        print("\nBest Model:")
        print(f"Model Name: {best_model_name}")
        print(f"Best F1 Score: {best_f1_score}")
        print(f"Best Parameters: {best_params}")

        return best_model, best_model_name, best_params



    def train_model(spark):

        df_batch = spark.read.csv("datasets/sarcasm.csv", header=True, inferSchema=True)  

        df_batch = df_batch.withColumn("text", lower(col("text")))
        
        df_batch = df_batch.filter(col("text").isNotNull() & (trim(col("text")) != ""))

        df_batch = df_batch.withColumn("class_num", 
            when(df_batch["class"] == "sarc", 1).when(df_batch["class"] == "notsarc", 0).otherwise(None)
        )

        df_batch = df_batch.filter(col("class_num").isNotNull())


        tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", pattern="\\W")
        remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
        word2Vec = Word2Vec(vectorSize=100, minCount=0, inputCol="filtered_tokens", outputCol="features")
        gbt = GBTClassifier(featuresCol="features", labelCol="class_num")

        pipe = Pipeline(stages=[tokenizer, remover, word2Vec, gbt])
        model = pipe.fit(df_batch)
        
        return model



    model = train_model(spark)


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
    
    predictions = model.transform(df)

    
    query = predictions.writeStream \
        .outputMode("append") \
        .format("console") \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    process_data()







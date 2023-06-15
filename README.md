# BigData_project
This project focuses on sentiment analysis using PySpark. It analyzes a dataset of tweets and classifies them into positive and negative sentiments. The project compares three machine learning models: Support Vector Machine (SVM), Logistic Regression, and Naive Bayes. Each model is tested with and without stemming.

# Dataset 
The dataset used in this project is the sentiment140 dataset It contains 1,600,000 tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 4 = positive) 
Link: https://www.kaggle.com/datasets/kazanova/sentiment140

# Preprocessing
Before training the machine learning models, the dataset undergoes several preprocessing steps. The steps include cleaning the data by removing mentions, URLs, and special characters. The text is converted to lowercase and leading/trailing whitespaces are removed. Tokenization is performed using a regular expression tokenizer to split the text into individual words. Stop words, such as common English words, are removed to reduce noise. Lastly, stemming is applied using the Snowball stemmer from the NLTK library to reduce words to their root form.

# Training and Evaluation
Three machine learning models are compared for sentiment analysis: Support Vector Machine (SVM), Logistic Regression, and Naive Bayes. The models are trained using the preprocessed dataset with and without stemming. The performance of each model is evaluated using accuracy as the metric.

# Classifying New Text
To classify new text using the trained Naive Bayes model, you can follow these steps:

1-Create a SparkSession.

2-Load the saved Naive Bayes model using the PipelineModel.load() function.

3-Provide the new text to be classified.

4-Create a DataFrame with the user text.

5-Perform the same data cleaning and preprocessing steps as done during training.

6-Make predictions using the loaded model on the preprocessed data.

7-Display the predicted labels.

#installation

1-Install the pyspark, nltk, and matplotlib libraries:
      
         !pip install pyspark nltk matplotlib


2-Import the necessary libraries
        
        from pyspark.sql import SparkSession
        from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
        from pyspark.ml.feature import StringIndexer
        from pyspark.ml.classification import LinearSVC
        from pyspark.ml.evaluation import MulticlassClassificationEvaluator
        from nltk.stem.snowball import SnowballStemmer
        from pyspark.sql.types import ArrayType, StringType
        from pyspark.sql.functions import udf
        
3-create a spark session

            spark = SparkSession.builder \
                .appName("SentimentAnalysis") \
                .getOrCreate()
4-load the dataset and select the target and tweet text columns

            data = spark.read.csv('/content/drive/MyDrive/training.1600000.processed.noemoticon.csv', header=False, inferSchema=True)
            data = data.selectExpr("_c0 as target", "_c5 as text")

5-perform data cleaning and preprocessing steps. Make sure to replace data with your actual dataset
       
        # Data cleaning
        data = data.withColumn("text", regexp_replace(col("text"), "@\S+|https?:\S+|http?:\S+|[^A-Za-z0-9\s]+", ""))
        data = data.withColumn("text", lower(col("text")))
        data = data.withColumn("text", trim(col("text")))

6-model without stemming Naive bayes example. 

        # Create a pipeline
          pipeline = Pipeline(stages=[
              RegexTokenizer(inputCol="text", outputCol="words", pattern=r"\W"),
              StopWordsRemover(inputCol="words", outputCol="filtered_words"),
              CountVectorizer(inputCol="filtered_words", outputCol="features"),
              StringIndexer(inputCol="target", outputCol="label"),
              NaiveBayes(featuresCol="features", labelCol="label", predictionCol="prediction")
          ])

          # Split the dataset into training and testing sets
          trainData, testData = data.randomSplit([0.8, 0.2], seed=42)

          # Train the pipeline
          model = pipeline.fit(trainData)

          # Save the trained model
          model.save("/content/sample_data/nbtest2.pickle")
          print('Model Saved')

          # Make predictions on the testing data
          predictions = model.transform(testData)


          # Evaluate the model's performance
          evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
          accuracy = evaluator.evaluate(predictions)
          print("Accuracy: {:.2f}%".format(accuracy * 100))
        
 7-model with stemming Naive Bayes example 
       
      stemmer = SnowballStemmer(language='english')

      # Tokenization
      tokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern=r"\W")

      # Remove stopwords from the tokenized words
      stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

      # Apply stemming using Snowball stemmer
      stemming_udf = udf(lambda words: [stemmer.stem(word) for word in words], ArrayType(StringType()))

      # Convert the stemmed words into numerical features using CountVectorizer
      count_vectorizer = CountVectorizer(inputCol="stemmed_words", outputCol="features")

      # Convert the target labels to numerical format
      label_indexer = StringIndexer(inputCol="target", outputCol="label")

      # Create a Naive Bayes classifier
      naive_bayes = NaiveBayes(featuresCol="features", labelCol="label", predictionCol="prediction")

      # Define the stages of the pipeline
      stages = [tokenizer, stopwords_remover, stemming_udf, count_vectorizer, label_indexer, naive_bayes]

      # Create a pipeline
      pipeline = Pipeline(stages=stages)

      # Fit the pipeline on the training data
      pipeline_model = pipeline.fit(trainData)

      # Make predictions on the testing data
      predictions = pipeline_model.transform(testData)

      # Evaluate the model's performance
      evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
      accuracy = evaluator.evaluate(predictions)
      print("Accuracy: {:.2f}%".format(accuracy * 100))
      
8-Classifying New Text. make sure to save the model and replace the path in load() function

      # Load the saved Naive Bayes model
      loadedModel = PipelineModel.load("/content/sample_data/nbtest2.pickle")

      # Assuming you have a new text provided by the user
      user_text = "text to clssify"

      # Create a DataFrame with the user text
      data = spark.createDataFrame([(user_text,)], ["text"])

      # Data cleaning
      data = data.withColumn("text", regexp_replace(col("text"), "@\S+|https?:\S+|http?:\S+|[^A-Za-z0-9\s]+", ""))
      data = data.withColumn("text", lower(col("text")))
      data = data.withColumn("text", trim(col("text")))

      # Tokenization
      tokenizer = RegexTokenizer(inputCol="text", outputCol="words_temp", pattern=r"\W")
      tokenized_data = tokenizer.transform(data).drop("words")

      # Rename the temporary column to "words"
      tokenized_data = tokenized_data.withColumnRenamed("words_temp", "words")

      # Remove stopwords from the tokenized words
      stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words_temp")
      filtered_data = stopwords_remover.transform(tokenized_data).drop("filtered_words")

      # Rename the temporary column to "filtered_words"
      filtered_data = filtered_data.withColumnRenamed("filtered_words_temp", "filtered_words")

      # Convert the filtered words into numerical features using CountVectorizer
      count_vectorizer = CountVectorizer(inputCol="filtered_words", outputCol="features")
      vectorized_data = count_vectorizer.fit(filtered_data).transform(filtered_data)

      # Make predictions using the loaded model
      predictions = loadedModel.transform(vectorized_data.select(['text']))

      # Show the predicted labels
      predictions.show()



# spark-fm
Factorization Machines is a general predictor like SVMs but is also able to estimate reliable parameters under very high sparsity. However, they are costly to scale to large amounts of data and large numbers of features. spark-fm is a parallel implementation of factorization machines based on Spark. Currently, we support mini-batch stochastic gradient descent to train the model. More optimization methods like parallel gradient descent, L-BFGS will be added soon.

# Examples
## Scala API
```scala
    val spark = SparkSession
        .builder()
        .appName("FactorizationMachinesExample")
        .master("local[4]")
        .getOrCreate()

    val train = spark.read.format("libsvm").load("data/a9a")
    val test = spark.read.format("libsvm").load("data/a9a.t")
    val trainer = new FactorizationMachines()
        .setTask("classification")
    val model = trainer.fit(train)
    val result = model.transform(test)
    val predictionAndLabel = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
    println("Accuracy: " + evaluator.evaluate(predictionAndLabel))
    spark.stop()
```

# Requirements
spark-fm is built against Spark 2.0.1.

# Build From Source
```scala
sbt package
```

# Licenses
spark-fm is available under Apache Licenses 2.0.

# Contact & Feedback
If you encounter bugs, feel free to submit an issue or pull request. Also you can mail to:
+ Chen Lin (m2linchen@gmail.com).

# Acknowledgement
Special thanks to Qian Huang(qianhuang@me.com) for his help to this project.

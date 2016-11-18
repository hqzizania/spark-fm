package org.apache.spark.ml

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.fm.FactorizationMachines
import org.apache.spark.sql.SparkSession

/**
  * An example for Factorization Machines.
  */
object FactorizationMachinesSuite {
  def main(args: Array[String]): Unit = {
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
  }
}

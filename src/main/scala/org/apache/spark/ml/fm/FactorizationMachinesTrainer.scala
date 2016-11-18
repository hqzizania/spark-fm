package org.apache.spark.ml.fm

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.optimization.ParallelGradientDescent
import org.apache.spark.mllib.linalg.{DenseVector, Vector => MLlibVector, Vectors => MLlibVectors}
import org.apache.spark.mllib.optimization.{Gradient, GradientDescent, Optimizer, Updater}
import org.apache.spark.mllib.linalg.VectorImplicits._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.util.Random

class FactorizationMachinesTrainer(val numFeatures: Int) {
  private var _task: String = "regression"
  private var _initStd: Double = 0.01d
  private var _weights: Vector = null
  private var _useBiasTerm: Boolean = true
  private var _useLinearTerms: Boolean = true
  private var _numFactors: Int = 8
  private var _regParams: (Double, Double, Double) = (0, 1e-3, 1e-4)
  private var _gradient: Gradient = new FactorizationMachinesGradient(
    _task, _useBiasTerm, _useLinearTerms, _numFactors, numFeatures, _regParams)
  private var _updater: Updater = new FactorizationMachinesUpdater(
    _useBiasTerm, _useLinearTerms, _numFactors, numFeatures, _regParams)
  private var optimizer: Optimizer = SGDOptimizer.setConvergenceTol(1e-4).setNumIterations(100)

  /**
    * Returns task
    */
  def getTask: String = _task

  /**
    * Sets task
    *
    * @param value task
    * @return trainer
    */
  def setTask(value: String): this.type = {
    _task = value
    this
  }

  /**
    * Returns weights
    */
  def getWeights: Vector = _weights

  /**
    * Sets weights
    *
    * @param value weights
    * @return trainer
    */
  def setWeights(value: Vector): this.type = {
    _weights = value
    this
  }

  /**
    * Returns initStd
    */
  def getInitStd: Double = _initStd

  /**
    * Sets initStd
    *
    * @param value initStd
    * @return trainer
    */
  def setInitStd(value: Double): this.type = {
    _initStd = value
    this
  }

  /**
    * Returns useBiasTerm
    */
  def getUseBiasTerm: Boolean = _useBiasTerm

  /**
    * Sets useBiasTerm
    *
    * @param value useBiasTerm
    * @return trainer
    */
  def setUseBiasTerm(value: Boolean): this.type = {
    _useBiasTerm = value
    this
  }

  /**
    * Returns useLinearTerms
    */
  def getUseLinearTerms: Boolean = _useLinearTerms

  /**
    * Sets useLinearTerms
    *
    * @param value useLinearTerms
    * @return trainer
    */
  def setUseLinearTerms(value: Boolean): this.type = {
    _useLinearTerms = value
    this
  }

  /**
    * Returns numFactors
    */
  def getNumFactors: Int = _numFactors

  /**
    * Sets numFactors
    *
    * @param value numFactors
    * @return trainer
    */
  def setNumFactors(value: Int): this.type = {
    _numFactors = value
    this
  }

  /**
    * Returns regParams
    */
  def getRegParams: (Double, Double, Double) = _regParams

  /**
    * Sets regParams
    *
    * @param value regParams
    * @return trainer
    */
  def setRegParams(value: (Double, Double, Double)): this.type = {
    _regParams = value
    this
  }

  /**
    * Sets the SGD optimizer
    *
    * @return SGD optimizer
    */
  def SGDOptimizer: GradientDescent = {
    val sgd = new GradientDescent(_gradient, _updater)
    optimizer = sgd
    sgd
  }

  /**
    * Sets the ParSGD optimizer
    *
    * @return ParSGD optimizer
    */
  def ParSGDOptimizer: ParallelGradientDescent = {
    val parSGD = new ParallelGradientDescent(_gradient, _updater)
    optimizer = parSGD
    parSGD
  }

  /**
    * Sets the updater
    *
    * @param value updater
    * @return trainer
    */
  def setUpdater(value: Updater): this.type = {
    _updater = value
    updateUpdater(value)
    this
  }

  /**
    * Sets the gradient
    *
    * @param value gradient
    * @return trainer
    */
  def setGradient(value: Gradient): this.type = {
    _gradient = value
    updateGradient(value)
    this
  }

  private[this] def updateGradient(gradient: Gradient): Unit = {
    optimizer match {
      case sgd: GradientDescent => sgd.setGradient(gradient)
      case parSGD: ParallelGradientDescent => parSGD.setGradient(gradient)
      case other => throw new UnsupportedOperationException(
        s"Only GradientDescent and ParallelGradientDescent are supported but got ${other.getClass}.")
    }
  }

  private[this] def updateUpdater(updater: Updater): Unit = {
    optimizer match {
      case sgd: GradientDescent => sgd.setUpdater(updater)
      case parSGD: ParallelGradientDescent => parSGD.setUpdater(updater)
      case other => throw new UnsupportedOperationException(
        s"Only GradientDescent and ParallelGradientDescent are supported but got ${other.getClass}.")
    }
  }

  /**
    * Trains Factorization Machines
    *
    * @param data RDD of labeledPoints
    * @return model
    */
  def train(data: RDD[LabeledPoint]): FactorizationMachinesTrainerModel = {
    val w = if (getWeights == null) {
      // TODO: will make a copy if vector is a subvector of BDV (see Vectors code)
      FactorizationMachinesTrainerModel(
          _initStd,
          _useBiasTerm,
          _useLinearTerms,
          _numFactors,
          numFeatures).weights
    } else {
      getWeights
    }
    // TODO: deprecate standard optimizer because it needs Vector
    val trainData = data.map { lp =>
      (lp.label, MLlibVectors.fromML(lp.features))
    }
    val handlePersistence = data.getStorageLevel == StorageLevel.NONE
    if (handlePersistence) {
      data.persist(StorageLevel.MEMORY_AND_DISK)
    }
    val newWeights = optimizer.optimize(trainData, w)
    if (handlePersistence) {
      trainData.unpersist()
    }
    FactorizationMachinesTrainerModel(newWeights)
  }
}

class FactorizationMachinesTrainerModel(val weights: Vector)
  extends Serializable {

  def predict(
      data: Vector,
      useBiasTerm: Boolean,
      useLinearTerms: Boolean,
      numFactors: Int,
      numFeatures: Int): (Double, Array[Double]) = {

    var prediction = if (useBiasTerm) {
      weights(weights.size - 1)
    }  else {
      0.0d
    }

    if (useBiasTerm) {
      val base = numFeatures * numFactors
      data.foreachActive {
        case (k, v) =>
          prediction += weights(base + k) * v
      }
    }

    val sum = Array.fill(numFactors)(0.0)
    for (i <- 0 until numFactors) {
      var sumSqr = 0.0
      data.foreachActive {
        case (k, v) =>
          val t = weights(k * numFactors + i) * v
          sum(i) += t
          sumSqr += t * t
      }
      prediction += (sum(i) * sum(i) - sumSqr) / 2
    }

    (prediction, sum)
  }
}

/**
  * Fabric for FactorizationMachinesTrainer models
  */
object FactorizationMachinesTrainerModel {

  /**
    * Creates a model from a weights
    *
    * @param weights weights
    * @return model
    */
  def apply(weights: Vector): FactorizationMachinesTrainerModel = {
    new FactorizationMachinesTrainerModel(weights)
  }

  /**
    * Creates a model given a standard deviation for initializing weights
    *
    * @param initStd a standard deviation for initializing weights
    * @return model
    */
  def apply(
      initStd: Double,
      useBiasTerm: Boolean,
      useLinearTerms: Boolean,
      numFactors: Int,
      numFeatures: Int): FactorizationMachinesTrainerModel = {
    val initMean = 0
    val weights = (useBiasTerm, useLinearTerms) match {
      case (true, true) =>
        Vectors.dense(Array.fill(numFeatures * numFactors)(Random.nextGaussian() * initStd + initMean) ++
          Array.fill(numFeatures + 1)(0.0))

      case (true, false) =>
        Vectors.dense(Array.fill(numFeatures * numFactors)(Random.nextGaussian() * initStd + initMean) ++
          Array(0.0))

      case (false, true) =>
        Vectors.dense(Array.fill(numFeatures * numFactors)(Random.nextGaussian() * initStd + initMean) ++
          Array.fill(numFeatures)(0.0))

      case (false, false) =>
        Vectors.dense(Array.fill(numFeatures * numFactors)(Random.nextGaussian() * initStd + initMean))
    }
    new FactorizationMachinesTrainerModel(weights)
  }
}

/**
  * :: DeveloperApi ::
  * Compute gradient and loss for a Least-squared loss function, as used in factorization machines.
  * For the detailed mathematical derivation, see the reference at
  * http://doi.acm.org/10.1145/2168752.2168771
  */
class FactorizationMachinesGradient(
    val task: String,
    val useBiasTerm: Boolean,
    val useLinearTerms: Boolean,
    val numFactors: Int,
    val numFeatures: Int,
    val regParams: (Double, Double, Double)) extends Gradient {

  override def compute(
      data: MLlibVector,
      label: Double,
      weights: MLlibVector): (MLlibVector, Double) = {
    val cumGradient = MLlibVectors.dense(Array.fill(weights.size)(0.0))
    val loss = compute(data, label, weights, cumGradient)
    (cumGradient, loss)
  }

  override def compute(
      data: MLlibVector,
      label: Double,
      weights: MLlibVector,
      cumGradient: MLlibVector): Double = {
    require(data.size == numFeatures)
    val (prediction, sum) = FactorizationMachinesTrainerModel(weights)
        .predict(data, useBiasTerm, useLinearTerms, numFactors, numFeatures)
    val multiplier = task match {
      case FactorizationMachines.Regression =>
        prediction - label
      case FactorizationMachines.Classification =>
        label * (1.0 / (1.0 + Math.exp(-label * prediction)) - 1.0)
    }

    cumGradient match {
      case vec: DenseVector =>
        val cumValues = vec.values

        if (useBiasTerm) {
          cumValues(cumValues.length - 1) += multiplier
        }

        if (useLinearTerms) {
          val pos = numFeatures * numFactors
          data.foreachActive {
            case (k, v) =>
              cumValues(pos + k) += v * multiplier
          }
        }

        data.foreachActive {
          case (k, v) =>
            val pos = k * numFactors
            for (f <- 0 until numFactors) {
              cumValues(pos + f) += (sum(f) * v - weights(pos + f) * v * v) * multiplier
            }
        }

      case _ =>
        throw new IllegalArgumentException(
          s"cumulateGradient only supports adding to a dense vector but got type ${cumGradient.getClass}.")
    }

    task match {
      case FactorizationMachines.Regression =>
        (prediction - label) * (prediction - label)
      case FactorizationMachines.Classification =>
        1 - Math.signum(prediction * label)
    }
  }

}


class FactorizationMachinesUpdater(
    useBiasTerm: Boolean,
    useLinearTerms: Boolean,
    numFactors: Int,
    numFeatures: Int,
    regParams: (Double, Double, Double)) extends Updater {

  override def compute(
      weightsOld: MLlibVector,
      gradient: MLlibVector,
      stepSize: Double,
      iter: Int,
      regParam: Double): (MLlibVector, Double) = {
    val r0 = regParams._1
    val r1 = regParams._2
    val r2 = regParams._3

    val thisIterStepSize = stepSize / math.sqrt(iter)
    val size = weightsOld.size

    val weightsNew = Array.fill(size)(0.0)
    var regVal = 0.0

    if (useBiasTerm) {
      weightsNew(size - 1) = weightsOld(size - 1) - thisIterStepSize * (gradient(size - 1) + r0 * weightsOld(size - 1))
      regVal += regParams._1 * weightsNew(size - 1) * weightsNew(size - 1)
    }

    if (useLinearTerms) {
      for (i <- numFeatures * numFactors until numFeatures * numFactors + numFeatures) {
        weightsNew(i) = weightsOld(i) - thisIterStepSize * (gradient(i) + r1 * weightsOld(i))
        regVal += r1 * weightsNew(i) * weightsNew(i)
      }
    }

    for (i <- 0 until numFeatures * numFactors) {
      weightsNew(i) = weightsOld(i) - thisIterStepSize * (gradient(i) + r2 * weightsOld(i))
      regVal += r2 * weightsNew(i) * weightsNew(i)
    }

    (Vectors.dense(weightsNew), regVal / 2)
  }
}

package org.apache.spark.ml.fm

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.shared.{HasMaxIter, HasStepSize, HasTol}
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.{PredictionModel, Predictor, PredictorParams}
import org.apache.spark.sql.Dataset

/** Params for Factorization Machines. */
private[ml] trait FactorizationMachinesParams extends PredictorParams
  with HasMaxIter with HasTol with HasStepSize {

  /**
    * The task that Factorization Machines are used to carry out.
    * Supported options: "regression" or "classification"(binary classification).
    * Default: "regression"
    *
    * @group expertParam
    */
  final val task: Param[String] = new Param[String](this, "task",
    "The task that Factorization Machines are used to carry out. Supported options: " +
      s"${FactorizationMachines.supportedTasks.mkString(",")}. (Default regression)",
    ParamValidators.inArray[String](FactorizationMachines.supportedTasks))

  /** @group expertGetParam */
  final def getTask: String = $(task)

  /**
    * The solver algorithm for optimization.
    * Supported options: "gd" (minibatch gradient descent) or "pargd"(parallel minibatch gradient descent).
    * Default: "gd"
    *
    * @group expertParam
    */
  final val solver: Param[String] = new Param[String](this, "solver",
    "The solver algorithm for optimization. Supported options: " +
      s"${FactorizationMachines.supportedSolvers.mkString(", ")}. (Default gd)",
    ParamValidators.inArray[String](FactorizationMachines.supportedSolvers))

  /** @group expertGetParam */
  final def getSolver: String = $(solver)

  /**
    * The initial weights of the model.
    *
    * @group expertParam
    */
  final val initialWeights: Param[Vector] = new Param[Vector](this, "initialWeights",
    "The initial weights of the model")

  /** @group expertGetParam */
  final def getInitialWeights: Vector = $(initialWeights)

  /**
    * The standard deviation for initializing weights.
    *
    * @group expertParam
    */
  final val initialStd: Param[Double] = new Param[Double](this,
    "initialStd", "The standard deviation for initializing weights")

  /** @group expertGetParam */
  final def getInitialStd: Double = $(initialStd)

  /**
    * Whether or not to use bias term.
    *
    * @group expertGetParam
    */
  final val useBiasTerm: Param[Boolean] = new Param[Boolean](this, "useBiasTerm",
    "Whether or not to use global bias term to train the model")

  /** @group expertGetParam */
  final def getUseBiasTerm: Boolean = $(useBiasTerm)

  /**
    * Whether or not to use linear terms.
    *
    * @group expertGetParam
    */
  final val useLinearTerms: Param[Boolean] = new Param[Boolean](this, "useLinearTerms",
    "Whether or not to use linear terms to train the model")

  /** @group expertGetParam */
  final def getUseLinearTerms: Boolean = $(useLinearTerms)

  /**
    * The number of factors that are used for pairwise interactions.
    *
    * @group expertGetParam
    */
  final val numFactors: Param[Int] = new Param[Int](this, "numFactors",
    "The number of factors that are used for pairwise interactions")

  /** @group expertGetParam */
  final def getNumFactors: Int = $(numFactors)

  /**
    * The regularization parameters of bias term, linear terms and pairwise interactions, respectively.
    *
    * @group expertGetParam
    */
  final val regParams: Param[(Double, Double, Double)] =
    new Param[(Double, Double, Double)](this, "regularization",
    "The regularization parameters of bias term, linear terms and pairwise interactions, respectively.")

  /** @group expertGetParam */
  final def getRegParams: (Double, Double, Double) = $(regParams)

  setDefault(maxIter -> 100, tol -> 1e-4, stepSize -> 0.03, initialStd -> 0.01,
    useBiasTerm -> true, useLinearTerms -> true, numFactors -> 8, regParams -> (0, 1e-3, 1e-4),
    solver -> FactorizationMachines.GD, task -> FactorizationMachines.Regression)
}

/**
  * Factorization Machines
  *
  * @param uid
  */
class FactorizationMachines(override val uid: String)
  extends Predictor[Vector, FactorizationMachines, FactorizationMachinesModel]
  with FactorizationMachinesParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("fm"))

  /**
    * Sets the value of param [[task]].
    * Default is "regression".
    *
    * @group expertSetParam
    */
  def setTask(value: String): this.type  = set(task, value)

  /**
    * Sets the value of param [[solver]].
    * Default is "gd".
    *
    * @group expertSetParam
    */
  def setSolver(value: String): this.type = set(solver, value)

  /**
    * Set the maximum number of iterations.
    * Default is 100.
    *
    * @group setParam
    */
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /**
    * Sets the value of param [[initialWeights]].
    *
    * @group expertSetParam
    */
  def setInitialWeights(value: Vector): this.type = set(initialWeights, value)

  /**
    * Sets the value of param [[stepSize]].
    * Default is 0.03.
    *
    * @group setParam
    */
  def setStepSize(value: Double): this.type = set(stepSize, value)

  /**
    * Sets the value of param [[initialStd]].
    * Default is 0.01.
    *
    * @group setParam
    */
  def setInitialStd(value: Double): this.type = set(initialStd, value)

  /**
    * Sets the value of param [[useBiasTerm]].
    * Default is true.
    *
    * @group setParam
    */
  def setUseBiasTerm(value: Boolean): this.type = set(useBiasTerm, value)

  /**
    * Sets the value of param [[useLinearTerms]].
    * Default is true.
    *
    * @group setParam
    */
  def setUseLinearTerms(value: Boolean): this.type = set(useLinearTerms, value)

  /**
    * Sets the value of param [[numFactors]].
    * Default is 8.
    *
    * @group setParam
    */
  def setNumFactors(value: Int): this.type = set(numFactors, value)

  /**
    * Sets the value of param [[regParams]].
    * Default is (0, 1e-3, 1e-4).
    *
    * @group setParam
    */
  def setRegParams(value: (Double, Double, Double)): this.type = set(regParams, value)

  override def copy(extra: ParamMap): FactorizationMachines = defaultCopy(extra)

  override protected def train(dataset: Dataset[_]): FactorizationMachinesModel = {
    val lpData = extractLabeledPoints(dataset)
    val numFeatures = lpData.first().features.size
    val trainer = new FactorizationMachinesTrainer(numFeatures)
    if (isDefined(initialWeights)) {
      trainer.setWeights($(initialWeights))
    } else {
      trainer.setInitStd($(initialStd))
    }
    if ($(solver) == FactorizationMachines.GD) {
      trainer.SGDOptimizer
        .setConvergenceTol($(tol))
        .setNumIterations($(maxIter))
        .setStepSize($(stepSize))
    } else {
      throw new IllegalArgumentException(
        s"The solver $solver is not supported by FactorizationMachines.")
    }
    val trainerModel = trainer.train(lpData)
    FactorizationMachinesModel(
      uid, trainerModel.weights, $(useBiasTerm), $(useLinearTerms), $(numFactors), numFeatures)
  }
}

object FactorizationMachines {
  /** String name for "gd" (minibatch gradient descent) solver. */
  private[ml] val GD = "gd"

  /** String name for "pargd" (parallel minibatch gradient descent) solver. */
  private[ml] val ParGD = "pargd"

  /** Set of solvers that MultilayerPerceptronClassifier supports. */
  private[ml] val supportedSolvers = Array(GD, ParGD)

  /** String name for "regression" task. */
  private[ml] val Regression = "regression"

  /** String name for "classification"(binary classification) task. */
  private[ml] val Classification = "classification"

  /** Set of tasks that Factorization Machines support. */
  private[ml] val supportedTasks = Array(Regression, Classification)
}

class FactorizationMachinesModel private[ml](
    override val uid: String,
    weights: Vector,
    useBiasTerm: Boolean,
    useLinearTerms: Boolean,
    numFactors: Int,
    numFeatures: Int)
  extends PredictionModel[Vector, FactorizationMachinesModel]
  with Serializable {

  private val trainerModel = FactorizationMachinesTrainerModel(weights)

  override protected def predict(features: Vector): Double = {
    val (prediction, _) = trainerModel.predict(features, useBiasTerm, useLinearTerms, numFactors, numFeatures)
    if (prediction > 0.5) {
      1.0d
    } else {
      -1.0d
    }
  }

  override def copy(extra: ParamMap): FactorizationMachinesModel = {
    copyValues(new FactorizationMachinesModel(
      uid, weights, useBiasTerm, useLinearTerms, numFactors, numFeatures), extra)
  }
}

object FactorizationMachinesModel {
  /**
    * Creates a model from a weights
    *
    * @param weights weights
    * @return model
    */
  def apply(
      uid: String,
      weights: Vector,
      useBiasTerm: Boolean,
      useLinearTerms: Boolean,
      numFactors: Int,
      numFeatures: Int): FactorizationMachinesModel = {
    new FactorizationMachinesModel(uid, weights, useBiasTerm, useLinearTerms, numFactors, numFeatures)
  }
}
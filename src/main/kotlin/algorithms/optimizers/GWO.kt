package com.technosudo.algorithms.optimizers

import com.technosudo.algorithms.fitness.FitnessFunction
import com.technosudo.algorithms.fitness.FitnessResult
import com.technosudo.evaluation.EvaluationMetrics
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.columnNames
import org.jetbrains.kotlinx.dataframe.api.select
import org.jetbrains.kotlinx.dataframe.api.toDataFrame
import java.io.File
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.random.Random

class GWO(
    override val populationSize: Int = 50,
    override val maxIterations: Int = 100,
    override val name: String = "Binary Grey Wolf Optimizer",
    private val logToCsv: Boolean = true,
    private val dataName: String = "Unnamed_Dataset",
    private val logPath: String = "src/main/kotlin/algorithms/logs/bgwo_${dataName}_logs.csv",
    private val mutationRate: Double = 0.05,
    private val minA: Double = 0.4,
    private val maxSolutions: Int = 1000
) : Optimizer {

    private fun sigmoid(x: Double): Double = 1.0 / (1.0 + exp(-x))
    private fun transfer(prob: Double): Int = if (Random.nextDouble() < prob) 1 else 0

    override fun optimize(dataset: DataFrame<*>, fitnessFunction: FitnessFunction): DataFrame<*> {
        require(populationSize >= 3) { "GWO requires a population size of at least 3." }

        val numFeatures = dataset.columnNames().size - 1
        require(numFeatures > 0) { "Dataset must have at least one feature." }

        var wolves = List(populationSize) {
            List(numFeatures) { if (Random.nextDouble() > 0.5) 1 else 0 }
        }

        var alpha = wolves[0]
        var beta = wolves[1]
        var delta = wolves[2]

        var alphaScore = Double.NEGATIVE_INFINITY
        var betaScore = Double.NEGATIVE_INFINITY
        var deltaScore = Double.NEGATIVE_INFINITY

        var totalEvaluations = 0
        var alphaMetrics = EvaluationMetrics(0.0, 0.0, 0.0, 0.0)
        var iter = 0

        if (logToCsv) {
            File(logPath).printWriter().use { out ->
                out.println("iteration,alpha_fitness,max_fitness,min_fitness,avg_fitness,alpha_accuracy,alpha_precision,alpha_recall,alpha_f1,features_selected,alpha_mask")
            }
        }

        println("Starting $name with a budget of $maxSolutions evaluations (Population: $populationSize).")

        while (totalEvaluations < maxSolutions) {
            val results: List<FitnessResult> = wolves.map { fitnessFunction.evaluateDetailed(dataset, it) }
            totalEvaluations += populationSize

            val fitnesses = results.map { it.fitness }

            wolves.zip(results).forEach { (wolf, result) ->
                val fitness = result.fitness
                when {
                    fitness > alphaScore -> {
                        delta = beta; deltaScore = betaScore
                        beta = alpha; betaScore = alphaScore
                        alpha = wolf; alphaScore = fitness
                        alphaMetrics = result.metrics
                    }
                    fitness > betaScore && wolf != alpha -> {
                        delta = beta; deltaScore = betaScore
                        beta = wolf; betaScore = fitness
                    }
                    fitness > deltaScore && wolf != alpha && wolf != beta -> {
                        delta = wolf; deltaScore = fitness
                    }
                }
            }

            val maxFitnessIter = fitnesses.maxOrNull() ?: Double.NaN
            val minFitnessIter = fitnesses.minOrNull() ?: Double.NaN
            val avgFitnessIter = fitnesses.average()

            val progress = totalEvaluations.toDouble() / maxSolutions
            val a = max(2.0 * (1.0 - progress), minA)

            wolves = wolves.map { wolf ->
                List(numFeatures) { i ->
                    val A1 = 2 * a * Random.nextDouble() - a
                    val C1 = 2 * Random.nextDouble()
                    val D_alpha = abs(C1 * alpha[i] - wolf[i])
                    val X1 = alpha[i] - A1 * D_alpha

                    val A2 = 2 * a * Random.nextDouble() - a
                    val C2 = 2 * Random.nextDouble()
                    val D_beta = abs(C2 * beta[i] - wolf[i])
                    val X2 = beta[i] - A2 * D_beta

                    val A3 = 2 * a * Random.nextDouble() - a
                    val C3 = 2 * Random.nextDouble()
                    val D_delta = abs(C3 * delta[i] - wolf[i])
                    val X3 = delta[i] - A3 * D_delta

                    val noise = Random.nextDouble(-0.3, 0.3)
                    val X_avg = (X1 + X2 + X3) / 3.0 + noise
                    transfer(sigmoid(X_avg))
                }
            }

            // Inject diversity if wolves become too similar
            if (wolves.distinct().size < populationSize / 3) {
                wolves = wolves.mapIndexed { i, wolf ->
                    if (i < populationSize / 3) List(numFeatures) { if (Random.nextDouble() > 0.5) 1 else 0 } else wolf
                }
            }

            wolves = wolves.map { wolf ->
                wolf.map { bit -> if (Random.nextDouble() < mutationRate) 1 - bit else bit }
            }

            val featuresSelected = alpha.count { it == 1 }
            println("Iteration ${iter + 1}: Fitness = ${"%.4f".format(alphaScore)}, Acc = ${"%.4f".format(alphaMetrics.accuracy)}, Features = $featuresSelected, Evals = $totalEvaluations/$maxSolutions")

            if (logToCsv) {
                File(logPath).appendText(
                    "${iter + 1},${"%.6f".format(alphaScore)},${"%.6f".format(maxFitnessIter)},${"%.6f".format(minFitnessIter)},${"%.6f".format(avgFitnessIter)}," +
                            "${"%.6f".format(alphaMetrics.accuracy)},${"%.6f".format(alphaMetrics.precision)},${"%.6f".format(alphaMetrics.recall)},${"%.6f".format(alphaMetrics.f1Score)}," +
                            "$featuresSelected,${alpha.joinToString("") { if (it == 1) "1" else "0" }}\n"
                )
            }

            iter++
            if (iter >= maxIterations || totalEvaluations >= maxSolutions) break
        }

        println("$name finished. Best fitness: ${"%.4f".format(alphaScore)}")
        println("Iterations run: $iter")
        println("Total solutions evaluated (actual): $totalEvaluations")

        val allColumnNames = dataset.columnNames()
        val featureNames = allColumnNames.dropLast(1)
        val targetColumnName = allColumnNames.last()
        val selectedFeatureNames = featureNames.filterIndexed { index, _ -> alpha.getOrNull(index) == 1 }
        val columnsToReturn = selectedFeatureNames + targetColumnName

        println("Returning a DataFrame with ${columnsToReturn.size} selected columns: $columnsToReturn")

        return dataset.select(*columnsToReturn.toTypedArray())
    }
}

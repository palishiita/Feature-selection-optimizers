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
    // maxIterations is now a failsafe, not the primary stopping condition.
    override val maxIterations: Int = 100,
    override val name: String = "Binary Grey Wolf Optimizer",
    private val logToCsv: Boolean = true,
    private val dataName: String = "Unnamed_Dataset",
    private val logPath: String = "src/main/kotlin/algorithms/logs/bgwo_${dataName}_logs.csv",
    private val mutationRate: Double = 0.015,
    private val minA: Double = 0.4,
    // The primary stopping condition is now the budget of total solutions to evaluate.
    private val maxSolutions: Int = 1000
) : Optimizer {

    private fun sigmoid(x: Double): Double = 1.0 / (1.0 + exp(-x))
    private fun transfer(prob: Double): Int = if (Random.nextDouble() < prob) 1 else 0

    override fun optimize(dataset: DataFrame<*>, fitnessFunction: FitnessFunction): DataFrame<*> {
        require(populationSize >= 3) { "GWO requires a population size of at least 3." }

        // The number of features is the column count minus one (for the target column).
        val numFeatures = dataset.columnNames().size - 1
        require(numFeatures > 0) { "Dataset must have at least one feature." }

        // --- Initialization ---
        var wolves = List(populationSize) {
            List(numFeatures) { if (Random.nextDouble() > 0.5) 1 else 0 }
        }

        // Initialize alpha, beta, and delta wolves with initial random positions
        var alpha = wolves.getOrElse(0) { emptyList() }
        var beta = wolves.getOrElse(1) { emptyList() }
        var delta = wolves.getOrElse(2) { emptyList() }

        var alphaScore = Double.NEGATIVE_INFINITY
        var betaScore = Double.NEGATIVE_INFINITY
        var deltaScore = Double.NEGATIVE_INFINITY

        // Initialize evaluation counter
        var totalEvaluations = 0

        if (logToCsv) {
            File(logPath).printWriter().use { out ->
                out.println(
                    "iteration,alpha_fitness,max_fitness,min_fitness,avg_fitness," +
                            "alpha_accuracy,alpha_precision,alpha_recall,alpha_f1,features_selected,alpha_mask"
                )
            }
        }

        // Updated startup message
        println("Starting $name with a budget of $maxSolutions evaluations (Population: $populationSize).")

        var alphaMetrics = EvaluationMetrics(0.0, 0.0, 0.0, 0.0)
        var iter = 0

        // --- Main Loop: Runs until evaluation budget is met ---
        while (totalEvaluations < maxSolutions) {

            // Evaluate the fitness of all wolves in the current population
            val results: List<FitnessResult> = wolves.map { fitnessFunction.evaluateDetailed(dataset, it) }
            totalEvaluations += populationSize // Increment evaluation counter
            val fitnesses = results.map { it.fitness }

            // Update alpha, beta, and delta wolves based on fitness
            wolves.zip(results).forEach { (wolf, result) ->
                val fitness = result.fitness
                when {
                    fitness > alphaScore -> {
                        delta = beta
                        deltaScore = betaScore
                        beta = alpha
                        betaScore = alphaScore
                        alpha = wolf
                        alphaScore = fitness
                        alphaMetrics = result.metrics
                    }
                    fitness > betaScore && wolf != alpha -> {
                        delta = beta
                        deltaScore = betaScore
                        beta = wolf
                        betaScore = fitness
                    }
                    fitness > deltaScore && wolf != alpha && wolf != beta -> {
                        delta = wolf
                        deltaScore = fitness
                    }
                }
            }

            val maxFitnessIter = fitnesses.maxOrNull() ?: Double.NaN
            val minFitnessIter = fitnesses.minOrNull() ?: Double.NaN
            val avgFitnessIter = fitnesses.average()

            // Update the 'a' parameter, which decreases linearly
            // Suggested change in the main loop
            val progress = totalEvaluations.toDouble() / maxSolutions
            val a = max(2.0 * (1.0 - progress), minA)
            // Update the position of each wolf
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

                    val X_avg = (X1 + X2 + X3) / 3.0
                    transfer(sigmoid(X_avg))
                }
            }

            // Apply mutation
            wolves = wolves.map { wolf ->
                wolf.map { bit -> if (Random.nextDouble() < mutationRate) 1 - bit else bit }
            }

            val featuresSelected = alpha.count { it == 1 }
            // Updated logging message to show evaluation progress
            println(
                "Iteration ${iter + 1}: Alpha Score = ${"%.4f".format(alphaScore)}, " +
                        "Acc = ${"%.4f".format(alphaMetrics.accuracy)}, Features = $featuresSelected, " +
                        "Evals = $totalEvaluations/$maxSolutions"
            )

            if (logToCsv) {
                File(logPath).appendText(
                    "${iter + 1},${"%.6f".format(alphaScore)}," +
                            "${"%.6f".format(maxFitnessIter)},${"%.6f".format(minFitnessIter)}," +
                            "${"%.6f".format(avgFitnessIter)}," +
                            "${"%.6f".format(alphaMetrics.accuracy)}," +
                            "${"%.6f".format(alphaMetrics.precision)}," +
                            "${"%.6f".format(alphaMetrics.recall)}," +
                            "${"%.6f".format(alphaMetrics.f1Score)}," +
                            "$featuresSelected,${alpha.joinToString("") { if (it == 1) "1" else "0" }}\n"
                )
            }

            iter++
            // Failsafe to prevent potential infinite loops
            if (iter >= maxIterations) {
                println("Warning: Exiting due to reaching the failsafe iteration limit ($maxIterations).")
                break
            }
            // Exit if budget is met before the next iteration
            if (totalEvaluations >= maxSolutions) break
        }

        println("$name finished. Best fitness: ${"%.4f".format(alphaScore)}")
        println("Iterations run: $iter")
        println("Total solutions evaluated (actual): $totalEvaluations")

        // --- MODIFICATION: Return DataFrame with selected columns ---
        // Get all column names from the original dataset.
        val allColumnNames = dataset.columnNames()
        // The feature names are all columns except the last one (assumed to be the target).
        val featureNames = allColumnNames.dropLast(1)
        // The target column name is the last column.
        val targetColumnName = allColumnNames.last()

        // Filter the feature names based on the final alpha wolf's position (the best solution).
        // A feature name is kept if its corresponding value in the mask is 1.
        val selectedFeatureNames = featureNames.filterIndexed { index, _ ->
            alpha.getOrNull(index) == 1
        }

        // The final list of columns for the new DataFrame includes the selected features and the target column.
        val columnsToReturn = selectedFeatureNames + targetColumnName

        println("Returning a DataFrame with ${columnsToReturn.size} selected columns: $columnsToReturn")

        // Use the `select` function to create a new DataFrame with only the desired columns.
        return dataset.select(*columnsToReturn.toTypedArray())
    }
}

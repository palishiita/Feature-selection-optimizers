package com.technosudo.algorithms.optimizers

import com.technosudo.algorithms.fitness.FitnessFunction
import com.technosudo.algorithms.fitness.FitnessResult
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.select
import org.jetbrains.kotlinx.dataframe.api.toDataFrame
import java.io.File
import java.util.Locale
import kotlin.math.exp
import kotlin.random.Random

class TLBO(
    override val populationSize: Int = 50,
    // maxIterations is now a failsafe, not the primary stopping condition.
    override val maxIterations: Int = 10,
    override val name: String = "Binary Teaching Learning Based Optimizer",
    private val logToCsv: Boolean = true,
    private val dataName: String = "Unnamed_Dataset",
    private val logPath: String = "src/main/kotlin/algorithms/logs/${dataName}_BTLBO_log.csv",
    private val mutationRate: Double = 0.015,
    // The primary stopping condition is now the budget of total solutions to evaluate.
    private val maxSolutions: Int = 500
) : Optimizer {

    private fun sigmoid(x: Double): Double = 1.0 / (1.0 + exp(-x))

    private fun transfer(prob: Double): Int = if (Random.nextDouble() < prob) 1 else 0

    private fun teacherPhase(
        population: List<BooleanArray>,
        fitnesses: List<Double>,
        dataset: DataFrame<*>,
        fitnessFunction: FitnessFunction
    ): Pair<List<BooleanArray>, List<Double>> {
        val numFeatures = population.first().size
        val teacherIndex = fitnesses.indices.maxByOrNull { fitnesses[it] } ?: 0
        val teacher = population[teacherIndex]

        val meanFeatures = DoubleArray(numFeatures) { j ->
            population.sumOf { if (it[j]) 1.0 else 0.0 } / population.size
        }

        val newPopulation = population.toMutableList()
        val newFitnesses = fitnesses.toMutableList()

        for (i in population.indices) {
            val originalLearner = population[i]
            val candidateLearner = BooleanArray(numFeatures) { j ->
                val tf = Random.nextInt(1, 3)
                val diff = (if (teacher[j]) 1.0 else 0.0) - tf * meanFeatures[j]
                val prob = sigmoid(diff)
                transfer(prob) == 1
            }

            val candidateFitness = fitnessFunction.evaluate(dataset, candidateLearner.map { if (it) 1 else 0 })
            if (candidateFitness > newFitnesses[i]) {
                newPopulation[i] = candidateLearner
                newFitnesses[i] = candidateFitness
            }
        }
        return newPopulation to newFitnesses
    }

    private fun learnerPhase(
        population: List<BooleanArray>,
        fitnesses: List<Double>,
        dataset: DataFrame<*>,
        fitnessFunction: FitnessFunction
    ): Pair<List<BooleanArray>, List<Double>> {
        val numFeatures = population.first().size
        val newPopulation = population.toMutableList()
        val newFitnesses = fitnesses.toMutableList()

        for (i in population.indices) {
            val peerIndex = (0 until population.size).filter { it != i }.random()
            val originalLearner = population[i]
            val peer = population[peerIndex]
            val originalFitness = fitnesses[i]
            val peerFitness = fitnesses[peerIndex]

            val (source, target) = if (peerFitness > originalFitness) {
                peer to originalLearner
            } else {
                originalLearner to peer
            }

            val candidateLearner = BooleanArray(numFeatures) { j ->
                val diff = (if (source[j]) 1.0 else 0.0) - (if (target[j]) 1.0 else 0.0)
                val prob = sigmoid(diff)
                (transfer(prob) == 1) xor originalLearner[j]
            }

            val candidateFitness = fitnessFunction.evaluate(dataset, candidateLearner.map { if (it) 1 else 0 })
            if (candidateFitness > newFitnesses[i]) {
                newPopulation[i] = candidateLearner
                newFitnesses[i] = candidateFitness
            }
        }
        return newPopulation to newFitnesses
    }

    /**
     * MODIFIED: This function now returns a Triple, with the third element being the
     * exact number of fitness evaluations performed during the mutation phase.
     */
    private fun mutateAndSelect(
        population: List<BooleanArray>,
        fitnesses: List<Double>,
        mutationRate: Double,
        dataset: DataFrame<*>,
        fitnessFunction: FitnessFunction
    ): Triple<List<BooleanArray>, List<Double>, Int> {
        val newPopulation = population.toMutableList()
        val newFitnesses = fitnesses.toMutableList()
        var evaluations = 0

        for (i in population.indices) {
            if (Random.nextDouble() < mutationRate) {
                val candidate = population[i].clone()
                val mutationPoint = Random.nextInt(candidate.size)
                candidate[mutationPoint] = !candidate[mutationPoint] // Flip a random bit

                val candidateFitness = fitnessFunction.evaluate(dataset, candidate.map { if (it) 1 else 0 })
                evaluations++ // Increment the counter for each evaluation
                if (candidateFitness > newFitnesses[i]) {
                    newPopulation[i] = candidate
                    newFitnesses[i] = candidateFitness
                }
            }
        }
        return Triple(newPopulation, newFitnesses, evaluations)
    }

    /**
     * MODIFIED: The main optimization loop now runs until the `maxSolutions` budget is
     * exhausted. The `maxIterations` parameter is only used as a failsafe.
     */
    override fun optimize(dataset: DataFrame<*>, fitnessFunction: FitnessFunction): DataFrame<*> {
        val numFeatures = dataset.columnNames().size - 1
        var population = List(populationSize) {
            BooleanArray(numFeatures) { Random.nextBoolean() }
        }

        var fitnessResults = population.map { fitnessFunction.evaluateDetailed(dataset, it.map { b -> if (b) 1 else 0 }) }
        var fitnesses = fitnessResults.map { it.fitness }
        var totalEvaluations = populationSize

        if (logToCsv) {
            File(logPath).printWriter().use { out ->
                out.println(
                    "iteration,best_fitness,max_fitness,min_fitness,avg_fitness," +
                            "best_accuracy,best_precision,best_recall,best_f1,features_selected,best_mask"
                )
            }
        }
        // Updated startup message
        println("Starting $name with a budget of $maxSolutions evaluations (Population: $populationSize).")

        var iter = 0
        // Loop until the evaluation budget is met
        while (totalEvaluations < maxSolutions) {

            // --- Teacher Phase ---
            val (popAfterTeacher, fitAfterTeacher) = teacherPhase(population, fitnesses, dataset, fitnessFunction)
            population = popAfterTeacher
            fitnesses = fitAfterTeacher
            totalEvaluations += populationSize
            if (totalEvaluations >= maxSolutions) break // Exit if budget is met

            // --- Learner Phase ---
            val (popAfterLearner, fitAfterLearner) = learnerPhase(population, fitnesses, dataset, fitnessFunction)
            population = popAfterLearner
            fitnesses = fitAfterLearner
            totalEvaluations += populationSize
            if (totalEvaluations >= maxSolutions) break // Exit if budget is met

            // --- Mutation Phase ---
            val (popAfterMutation, fitAfterMutation, mutationEvals) = mutateAndSelect(population, fitnesses, mutationRate, dataset, fitnessFunction)
            population = popAfterMutation
            fitnesses = fitAfterMutation
            totalEvaluations += mutationEvals // Use the exact count from the function
            if (totalEvaluations >= maxSolutions) break // Exit if budget is met

            // --- Update and Log ---
            val bestCurrentIndex = fitnesses.indices.maxByOrNull { fitnesses[it] }!!
            val bestResult = fitnessFunction.evaluateDetailed(dataset, population[bestCurrentIndex].map { if (it) 1 else 0 })

            val bestFitness = bestResult.fitness
            val bestMetrics = bestResult.metrics
            val bestSolution = population[bestCurrentIndex]

            val maxFitnessIter = fitnesses.maxOrNull() ?: Double.NaN
            val minFitnessIter = fitnesses.minOrNull() ?: Double.NaN
            val avgFitnessIter = fitnesses.average()
            val featuresSelected = bestSolution.count { it }

            // Updated logging message to show evaluation progress
            println(
                "Iteration ${iter + 1}: Best Fitness = ${"%.4f".format(bestFitness)}, " +
                        "Acc = ${"%.4f".format(bestMetrics.accuracy)}, Features = $featuresSelected, " +
                        "Evals = $totalEvaluations/$maxSolutions"
            )

            if (logToCsv) {
                File(logPath).appendText(
                    String.format(
                        Locale.US,
                        "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%s\n",
                        iter + 1, bestFitness, maxFitnessIter, minFitnessIter, avgFitnessIter,
                        bestMetrics.accuracy, bestMetrics.precision, bestMetrics.recall, bestMetrics.f1Score,
                        featuresSelected, bestSolution.joinToString("") { if (it) "1" else "0" }
                    )
                )
            }
            iter++

            // Failsafe to prevent potential infinite loops
            if (iter >= maxIterations) {
                println("Warning: Exiting due to reaching the failsafe iteration limit ($maxIterations).")
                break
            }
        }

        val finalBestIndex = fitnesses.indices.maxByOrNull { fitnesses[it] }!!
        val finalBestSolution = population[finalBestIndex]

        println("$name finished. Best fitness: ${"%.4f".format(fitnesses[finalBestIndex])}")
        println("Iterations run: $iter")
        println("Total solutions evaluated (actual): $totalEvaluations")
        println("PRUNE FINAL SOLUTION ${finalBestSolution.joinToString { if (it) "1" else "0" }}")

        // Get all column names from the original dataset.
        val allColumnNames = dataset.columnNames()
        // The feature names are all columns except the last one (assumed to be the target).
        val featureNames = allColumnNames.dropLast(1)
        // The target column name is the last column.
        val targetColumnName = allColumnNames.last()

        // Filter the feature names based on the finalBestSolution mask.
        // A feature name is kept if its corresponding value in the mask is 'true'.
        val selectedFeatureNames = featureNames.filterIndexed { index, _ ->
            finalBestSolution[index]
        }

        // The final list of columns for the new DataFrame includes the selected features and the target column.
        val columnsToReturn = selectedFeatureNames + targetColumnName

        println("Returning a DataFrame with ${columnsToReturn.size} selected columns: $columnsToReturn")

        // Use the `select` function to create a new DataFrame with only the desired columns.
        return dataset.select(*columnsToReturn.toTypedArray())
    }
}
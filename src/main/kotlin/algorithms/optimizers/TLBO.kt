package com.technosudo.algorithms.optimizers

import com.technosudo.algorithms.fitness.FitnessFunction
import com.technosudo.algorithms.fitness.FitnessResult
import kotlinx.coroutines.*
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.toDataFrame
import java.io.File
import java.util.Locale
import kotlin.math.exp
import kotlin.random.Random

class TLBO(
    override val populationSize: Int = 10,
    override val maxIterations: Int = 30,
    override val name: String = "Binary Teaching Learning Based Optimizer",
    private val logToCsv: Boolean = true,
    private val dataName: String = "Unnamed_Dataset",
    private val logPath: String = "src/main/kotlin/algorithms/logs/${dataName}_BTLBO_log.csv",
    private val mutationRate: Double = 0.02
) : Optimizer {

    private fun sigmoid(x: Double): Double = 1.0 / (1.0 + exp(-x))

    private fun transfer(prob: Double): Int = if (Random.nextDouble() < prob) 1 else 0

    private fun teacherPhase(
        population: List<BooleanArray>,
        fitnesses: List<Double>,
        numFeatures: Int
    ): List<BooleanArray> {
        val teacherIndex = fitnesses.indices.maxByOrNull { fitnesses[it] } ?: 0
        val teacher = population[teacherIndex]

        val meanFeatures = DoubleArray(numFeatures) { j ->
            population.sumOf { if (it[j]) 1.0 else 0.0 } / population.size
        }

        return population.map { learner ->
            BooleanArray(numFeatures) { i ->
                val TF = if (Random.nextDouble() < 0.5) 1 else 2
                val diff = (if (teacher[i]) 1.0 else 0.0) - TF * meanFeatures[i]
                val prob = sigmoid(diff)
                transfer(prob) == 1
            }
        }
    }

    private fun learnerPhase(
        population: List<BooleanArray>,
        fitnesses: List<Double>,
        numFeatures: Int
    ): List<BooleanArray> {
        val newPopulation = population.toMutableList()

        for (i in population.indices) {
            val learner = population[i]
            var peerIndex: Int
            do {
                peerIndex = Random.nextInt(population.size)
            } while (peerIndex == i)

            val peer = population[peerIndex]
            val learnerFitness = fitnesses[i]
            val peerFitness = fitnesses[peerIndex]

            val newLearner = BooleanArray(numFeatures) { j ->
                val diff = if (peerFitness > learnerFitness)
                    (if (peer[j]) 1 else 0) - (if (learner[j]) 1 else 0)
                else
                    (if (learner[j]) 1 else 0) - (if (peer[j]) 1 else 0)

                val prob = sigmoid(diff.toDouble())
                transfer(prob) == 1
            }

            if (!learner.contentEquals(newLearner)) {
                newPopulation[i] = newLearner
            }
        }
        return newPopulation
    }

    private fun mutatePopulation(
        population: List<BooleanArray>,
        mutationRate: Double
    ): List<BooleanArray> {
        return population.map { individual ->
            BooleanArray(individual.size) { i ->
                if (Random.nextDouble() < mutationRate) !individual[i] else individual[i]
            }
        }
    }

    private suspend fun evaluatePopulationParallel(
        dataset: DataFrame<*>,
        population: List<BooleanArray>,
        fitnessFunction: FitnessFunction
    ): List<FitnessResult> = coroutineScope {
        population.map { mask ->
            async(Dispatchers.Default) {
                fitnessFunction.evaluateDetailed(dataset, mask.toList().map { if (it) 1 else 0 })
            }
        }.awaitAll()
    }

    override fun optimize(dataset: DataFrame<*>, fitnessFunction: FitnessFunction): DataFrame<*> {
        val numFeatures = dataset.columnNames().size

        var population = List(populationSize) {
            BooleanArray(numFeatures) { Random.nextBoolean() }
        }

        var results = runBlocking { evaluatePopulationParallel(dataset, population, fitnessFunction) }
        var fitnesses = results.map { it.fitness }

        var bestIndex = fitnesses.indices.maxByOrNull { fitnesses[it] } ?: 0
        var bestSolution = population[bestIndex]
        var bestFitness = fitnesses[bestIndex]
        var bestMetrics = results[bestIndex].metrics

        val logBuffer = StringBuilder()
        if (logToCsv) {
            File(logPath).printWriter().use { out ->
                out.println(
                    "iteration|best_fitness|max_fitness|min_fitness|avg_fitness," +
                            "best_accuracy|best_precision|best_recall|best_f1|features_selected|best_mask"
                )
            }
        }

        println("Starting $name with $populationSize learners and $maxIterations iterations.")

        repeat(maxIterations) { iter ->
            population = teacherPhase(population, fitnesses, numFeatures)
            results = runBlocking { evaluatePopulationParallel(dataset, population, fitnessFunction) }
            fitnesses = results.map { it.fitness }

            population = learnerPhase(population, fitnesses, numFeatures)
            results = runBlocking { evaluatePopulationParallel(dataset, population, fitnessFunction) }
            fitnesses = results.map { it.fitness }

            population = mutatePopulation(population, mutationRate)
            results = runBlocking { evaluatePopulationParallel(dataset, population, fitnessFunction) }
            fitnesses = results.map { it.fitness }

            val currentBestIndex = fitnesses.indices.maxByOrNull { fitnesses[it] } ?: 0
            val currentBestFitness = fitnesses[currentBestIndex]
            if (currentBestFitness > bestFitness) {
                bestFitness = currentBestFitness
                bestSolution = population[currentBestIndex]
                bestMetrics = results[currentBestIndex].metrics
            }

            val maxFitnessIter = fitnesses.maxOrNull() ?: Double.NaN
            val minFitnessIter = fitnesses.minOrNull() ?: Double.NaN
            val avgFitnessIter = fitnesses.average()
            val featuresSelected = bestSolution.count { it }

            println(
                "Iteration ${iter + 1}/$maxIterations: Best Fitness = ${"%.4f".format(Locale.US, bestFitness)}, " +
                        "Max = ${"%.4f".format(Locale.US, maxFitnessIter)}, Min = ${"%.4f".format(Locale.US, minFitnessIter)}, " +
                        "Avg = ${"%.4f".format(Locale.US, avgFitnessIter)}, " +
                        "Acc = ${"%.4f".format(Locale.US, bestMetrics.accuracy)}, " +
                        "Prec = ${"%.4f".format(Locale.US, bestMetrics.precision)}, " +
                        "Rec = ${"%.4f".format(Locale.US, bestMetrics.recall)}, " +
                        "F1 = ${"%.4f".format(Locale.US, bestMetrics.f1Score)}, Features Selected = $featuresSelected"
            )

            if (logToCsv) {
                logBuffer.append(
                    String.format(
                        Locale.US,
                        "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%s\n",
                        iter + 1,
                        bestFitness,
                        maxFitnessIter,
                        minFitnessIter,
                        avgFitnessIter,
                        bestMetrics.accuracy,
                        bestMetrics.precision,
                        bestMetrics.recall,
                        bestMetrics.f1Score,
                        featuresSelected,
                        bestSolution.joinToString("") { if (it) "1" else "0" }
                    )
                )
            }
        }

        if (logToCsv) {
            File(logPath).appendText(logBuffer.toString())
        }

        println("$name finished. Best fitness: ${"%.4f".format(bestFitness)}")
        return listOf(bestSolution.map { if (it) 1 else 0 }).toDataFrame()
    }
}

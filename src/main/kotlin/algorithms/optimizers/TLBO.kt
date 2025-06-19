package com.technosudo.algorithms.optimizers

import com.technosudo.algorithms.fitness.FitnessFunction
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.toDataFrame
import java.io.File
import java.util.Locale
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

    // Transfer function to convert continuous update to binary (0/1)
    private fun transfer(prob: Double): Int = if (Random.nextDouble() < prob) 1 else 0

    // Teacher Phase: improve learners by moving towards the teacher (best solution)
    private fun teacherPhase(
        population: List<List<Int>>,
        fitnesses: List<Double>,
        numFeatures: Int
    ): List<List<Int>> {
        val teacherIndex = fitnesses.indices.maxByOrNull { fitnesses[it] } ?: 0
        val teacher = population[teacherIndex]

        // Calculate mean of each feature across population
        val meanFeatures = DoubleArray(numFeatures) { featureIndex ->
            population.sumOf { it[featureIndex].toDouble() } / population.size
        }

        return population.map { learner ->
            List(numFeatures) { i ->
                // Teaching factor TF randomly 1 or 2
                val TF = if (Random.nextDouble() < 0.5) 1 else 2
                // Update rule: learner moves towards teacher knowledge
                val diff = teacher[i] - TF * meanFeatures[i]
                // Use sigmoid to map to probability of selecting feature
                val prob = 1.0 / (1.0 + kotlin.math.exp(-diff))
                transfer(prob)
            }
        }
    }

    // Learner Phase: learners learn from each other by pairwise interaction
    private fun learnerPhase(
        population: List<List<Int>>,
        fitnesses: List<Double>,
        numFeatures: Int
    ): List<List<Int>> {
        val newPopulation = population.toMutableList()

        for (i in population.indices) {
            val learner = population[i]
            // Select a random peer different from the learner
            var peerIndex: Int
            do {
                peerIndex = Random.nextInt(population.size)
            } while (peerIndex == i)
            val peer = population[peerIndex]

            val learnerFitness = fitnesses[i]
            val peerFitness = fitnesses[peerIndex]

            val newLearner = List(numFeatures) { j ->
                if (peerFitness > learnerFitness) {
                    // Move learner towards peer
                    val diff = peer[j] - learner[j]
                    val prob = 1.0 / (1.0 + kotlin.math.exp((-diff).toDouble()))
                    transfer(prob)
                } else {
                    // Move learner away from peer
                    val diff = learner[j] - peer[j]
                    val prob = 1.0 / (1.0 + kotlin.math.exp((-diff).toDouble()))
                    transfer(prob)
                }
            }

            newPopulation[i] = newLearner
        }

        return newPopulation
    }

    // Mutation: flip each bit with probability = mutationRate
    private fun mutatePopulation(
        population: List<List<Int>>,
        mutationRate: Double
    ): List<List<Int>> {
        return population.map { individual ->
            individual.map { gene ->
                if (Random.nextDouble() < mutationRate) 1 - gene else gene
            }
        }
    }

    override fun optimize(dataset: DataFrame<*>, fitnessFunction: FitnessFunction): DataFrame<*> {
        val numFeatures = dataset.columnNames().size

        // Initialize population randomly (binary vectors)
        var population = List(populationSize) {
            List(numFeatures) { if (Random.nextDouble() > 0.5) 1 else 0 }
        }

        var results = population.map { fitnessFunction.evaluateDetailed(dataset, it) }
        var fitnesses = results.map { it.fitness }

        var bestIndex = fitnesses.indices.maxByOrNull { fitnesses[it] } ?: 0
        var bestSolution = population[bestIndex]
        var bestFitness = fitnesses[bestIndex]
        var bestMetrics = results[bestIndex].metrics

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
            // Teacher phase
            population = teacherPhase(population, fitnesses, numFeatures)
            results = population.map { fitnessFunction.evaluateDetailed(dataset, it) }
            fitnesses = results.map { it.fitness }

            // Learner phase
            population = learnerPhase(population, fitnesses, numFeatures)
            results = population.map { fitnessFunction.evaluateDetailed(dataset, it) }
            fitnesses = results.map { it.fitness }

            // Mutation phase
            population = mutatePopulation(population, mutationRate)
            results = population.map { fitnessFunction.evaluateDetailed(dataset, it) }
            fitnesses = results.map { it.fitness }

            // Update best solution and metrics
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
            val featuresSelected = bestSolution.count { it == 1 }

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
                File(logPath).appendText(
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
                        bestSolution.joinToString("")
                    )
                )
            }
        }

        println("$name finished. Best fitness: ${"%.4f".format(bestFitness)}")
        return listOf(bestSolution).toDataFrame()
    }
}

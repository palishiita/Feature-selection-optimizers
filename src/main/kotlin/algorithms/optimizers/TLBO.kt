package com.technosudo.algorithms.optimizers

import com.technosudo.algorithms.fitness.FitnessFunction
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.toDataFrame
import java.io.File
import kotlin.random.Random

class TLBO(
    override val populationSize: Int = 10,
    override val maxIterations: Int = 30,
    override val name: String = "Binary Teaching Learning Based Optimizer",
    private val logToCsv: Boolean = true,
    private val logPath: String = "btlbo_log.csv"
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

            // Accept new learner if fitness improves
            // Fitness evaluation will be done outside this function
            newPopulation[i] = newLearner
        }

        return newPopulation
    }

    override fun optimize(dataset: DataFrame<*>, fitnessFunction: FitnessFunction): DataFrame<*> {
        val numFeatures = dataset.columnNames().size

        // Initialize population randomly (binary vectors)
        var population = List(populationSize) {
            List(numFeatures) { if (Random.nextDouble() > 0.5) 1 else 0 }
        }

        // Evaluate initial fitnesses
        var fitnesses = population.map { fitnessFunction.evaluate(dataset, it) }

        // Track best solution
        var bestIndex = fitnesses.indices.maxByOrNull { fitnesses[it] } ?: 0
        var bestSolution = population[bestIndex]
        var bestFitness = fitnesses[bestIndex]

        if (logToCsv) {
            File(logPath).printWriter().use { out ->
                out.println("iteration,best_fitness,features_selected,best_mask")
            }
        }

        println("Starting $name with $populationSize learners and $maxIterations iterations.")

        repeat(maxIterations) { iter ->
            // Teacher phase
            population = teacherPhase(population, fitnesses, numFeatures)
            fitnesses = population.map { fitnessFunction.evaluate(dataset, it) }

            // Learner phase
            population = learnerPhase(population, fitnesses, numFeatures)
            fitnesses = population.map { fitnessFunction.evaluate(dataset, it) }

            // Update best solution
            val currentBestIndex = fitnesses.indices.maxByOrNull { fitnesses[it] } ?: 0
            val currentBestFitness = fitnesses[currentBestIndex]
            if (currentBestFitness > bestFitness) {
                bestFitness = currentBestFitness
                bestSolution = population[currentBestIndex]
            }

            val featuresSelected = bestSolution.count { it == 1 }
            println("Iteration ${iter + 1}/$maxIterations: Best Fitness = ${"%.4f".format(bestFitness)}, Features Selected = $featuresSelected")

            if (logToCsv) {
                File(logPath).appendText(
                    "${iter + 1},${"%.6f".format(bestFitness)},$featuresSelected,${bestSolution.joinToString("")}\n"
                )
            }
        }

        println("$name finished. Best fitness: ${"%.4f".format(bestFitness)}")
        return listOf(bestSolution).toDataFrame()
    }
}

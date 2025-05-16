package com.technosudo.algorithms.optimizers

import com.technosudo.algorithms.fitness.FitnessFunction
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.select

class GWO(
    override val populationSize: Int = 30,
    override val maxIterations: Int = 100,
) : Optimizer {

    override val name: String = "Binary Grey Wolf Optimizer"

    override fun optimize(
        dataset: DataFrame<*>,
        fitnessFunction: FitnessFunction
    ): DataFrame<*> {
        val numFeatures = dataset.ncol
        // Initialize population: each wolf is a binary vector
        var population = Array(populationSize) { BooleanArray(numFeatures) { Math.random() < 0.5 } }
        var fitness = DoubleArray(populationSize) { i -> fitnessFunction.evaluate(dataset, population[i].map { if (it) 1 else 0 }) }

        // Identify alpha, beta, delta wolves (best, 2nd, 3rd best)
        fun updateLeaders(): Triple<Int, Int, Int> {
            val sorted = fitness.withIndex().sortedBy { it.value }
            return Triple(sorted[0].index, sorted[1].index, sorted[2].index)
        }

        repeat(maxIterations) { iter ->
            val a = 2.0 - 2.0 * iter / maxIterations // linearly decreases from 2 to 0
            val (alphaIdx, betaIdx, deltaIdx) = updateLeaders()
            val alpha = population[alphaIdx]
            val beta = population[betaIdx]
            val delta = population[deltaIdx]

            for (i in 0 until populationSize) {
                val wolf = population[i]
                val newWolf = BooleanArray(numFeatures)
                for (d in 0 until numFeatures) {
                    // GWO position update (binary version using sigmoid transfer)
                    val r1 = Math.random()
                    val r2 = Math.random()
                    val A1 = 2 * a * r1 - a
                    val C1 = 2 * r2
                    val D_alpha = Math.abs(C1 * (if (alpha[d]) 1.0 else 0.0) - (if (wolf[d]) 1.0 else 0.0))
                    val X1 = (if (alpha[d]) 1.0 else 0.0) - A1 * D_alpha

                    val r3 = Math.random()
                    val r4 = Math.random()
                    val A2 = 2 * a * r3 - a
                    val C2 = 2 * r4
                    val D_beta = Math.abs(C2 * (if (beta[d]) 1.0 else 0.0) - (if (wolf[d]) 1.0 else 0.0))
                    val X2 = (if (beta[d]) 1.0 else 0.0) - A2 * D_beta

                    val r5 = Math.random()
                    val r6 = Math.random()
                    val A3 = 2 * a * r5 - a
                    val C3 = 2 * r6
                    val D_delta = Math.abs(C3 * (if (delta[d]) 1.0 else 0.0) - (if (wolf[d]) 1.0 else 0.0))
                    val X3 = (if (delta[d]) 1.0 else 0.0) - A3 * D_delta

                    val X_avg = (X1 + X2 + X3) / 3.0
                    // Binary transfer function (sigmoid)
                    val S = 1.0 / (1.0 + Math.exp(-X_avg))
                    newWolf[d] = Math.random() < S
                }
                population[i] = newWolf
                fitness[i] = fitnessFunction.evaluate(dataset, newWolf.map { if (it) 1 else 0 })
            }
        }
        // Return the best solution as a DataFrame with selected features
        val bestIdx = fitness.withIndex().minByOrNull { it.value }!!.index
        val bestMask = population[bestIdx]
        val selectedCols = dataset.columns().filterIndexed { idx, _ -> bestMask[idx] }
        return dataset.select { selectedCols }
    }
}
package com.technosudo.algorithms.optimizers

import com.technosudo.algorithms.fitness.FitnessFunction
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.toDataFrame
import java.io.File
import kotlin.math.abs
import kotlin.math.exp
import kotlin.random.Random

class GWO(
    override val populationSize: Int = 10,
    override val maxIterations: Int = 30,
    override val name: String = "Binary Grey Wolf Optimizer",
    private val logToCsv: Boolean = true,
    private val logPath: String = "bgwo_log2.csv"
) : Optimizer {

    private fun sigmoid(x: Double): Double = 1.0 / (1.0 + exp(-x))
    private fun transfer(prob: Double): Int = if (Random.nextDouble() < prob) 1 else 0

    override fun optimize(dataset: DataFrame<*>, fitnessFunction: FitnessFunction): DataFrame<*> {
        val numFeatures = dataset.columnNames().size

        var wolves = List(populationSize) {
            List(numFeatures) { if (Random.nextDouble() > 0.5) 1 else 0 }
        }

        var alpha = wolves[0]
        var beta = wolves[1]
        var delta = wolves[2]

        var alphaScore = Double.MIN_VALUE
        var betaScore = Double.MIN_VALUE
        var deltaScore = Double.MIN_VALUE

        if (logToCsv) {
            File(logPath).printWriter().use { out ->
                out.println("iteration,alpha_fitness,features_selected,alpha_mask")
            }
        }

        println("Starting $name with $populationSize wolves and $maxIterations iterations.")

        repeat(maxIterations) { iter ->
            wolves = wolves.map { wolf ->
                val fitness = fitnessFunction.evaluate(dataset, wolf)

                when {
                    fitness > alphaScore -> {
                        delta = beta
                        deltaScore = betaScore
                        beta = alpha
                        betaScore = alphaScore
                        alpha = wolf
                        alphaScore = fitness
                    }
                    fitness > betaScore -> {
                        delta = beta
                        deltaScore = betaScore
                        beta = wolf
                        betaScore = fitness
                    }
                    fitness > deltaScore -> {
                        delta = wolf
                        deltaScore = fitness
                    }
                }

                wolf
            }

            val a = 2.0 * (1.0 - iter.toDouble() / maxIterations)

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

            val featuresSelected = alpha.count { it == 1 }
            println("Iteration ${iter + 1}/$maxIterations: Alpha Score = ${"%.4f".format(alphaScore)}, Features Selected = $featuresSelected")

            if (logToCsv) {
                File(logPath).appendText(
                    "${iter + 1},${"%.6f".format(alphaScore)},$featuresSelected,${alpha.joinToString("")}\n"
                )
            }
        }

        println("$name finished. Best fitness: ${"%.4f".format(alphaScore)}")
        return listOf(alpha).toDataFrame()
    }
}

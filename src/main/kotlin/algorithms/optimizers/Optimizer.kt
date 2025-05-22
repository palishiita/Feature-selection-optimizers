package com.technosudo.algorithms.optimizers

import com.technosudo.algorithms.fitness.FitnessFunction
import org.jetbrains.kotlinx.dataframe.DataFrame

interface Optimizer {
    val name: String
    val populationSize: Int
    val maxIterations: Int

    fun optimize(
        dataset: DataFrame<*>,
        fitnessFunction: FitnessFunction
    ): DataFrame<*>
}
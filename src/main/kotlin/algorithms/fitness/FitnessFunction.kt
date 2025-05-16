package com.technosudo.algorithms.fitness

import org.jetbrains.kotlinx.dataframe.DataFrame

interface FitnessFunction {
    fun evaluate(
        dataset: DataFrame<*>,
        featureMask: List<Int>
    ): Double
}
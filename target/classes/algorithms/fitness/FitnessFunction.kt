package com.technosudo.algorithms.fitness

import org.jetbrains.kotlinx.dataframe.DataFrame
import com.technosudo.evaluation.EvaluationMetrics

interface FitnessFunction {
    fun evaluate(
        dataset: DataFrame<*>,
        featureMask: List<Int>
    ): Double

    fun evaluateDetailed(
        dataset: DataFrame<*>,
        featureMask: List<Int>
    ): FitnessResult {
        val fitness = evaluate(dataset, featureMask)
        return FitnessResult(fitness, EvaluationMetrics(Double.NaN, Double.NaN, Double.NaN, Double.NaN))
    }
}
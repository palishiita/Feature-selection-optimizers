package com.technosudo.evaluation

import org.jetbrains.kotlinx.dataframe.DataColumn
import org.jetbrains.kotlinx.dataframe.DataFrame

interface ClassifierWrapper {
    fun fit(train: DataFrame<*>, target: DataFrame<*>): ClassifierWrapper
    fun predict(test: DataFrame<*>): List<Double>
    fun evaluate(test: DataFrame<*>, target: DataFrame<*>): EvaluationMetrics
}

data class EvaluationMetrics(
    val accuracy: Double,
    val precision: Double,
    val recall: Double,
    val f1Score: Double,
    //val confusionMatrix: Map<Pair<Any?, Any?>, Int>
)
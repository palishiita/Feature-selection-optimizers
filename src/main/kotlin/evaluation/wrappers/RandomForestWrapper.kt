package com.technosudo.evaluation.wrappers

import com.technosudo.evaluation.ClassifierWrapper
import com.technosudo.evaluation.EvaluationMetrics
import org.jetbrains.kotlinx.dataframe.DataColumn
import org.jetbrains.kotlinx.dataframe.DataFrame
import smile.classification.RandomForest

class RandomForestWrapper : ClassifierWrapper {

    lateinit var model: RandomForest

    override fun fit(trainData: DataFrame<*>, target: DataFrame<*>): ClassifierWrapper {
//        RandomForest.fit(model, trainData, targetColumn)
        return this
    }

    override fun predict(testData: DataFrame<*>): List<Double> {
        TODO("Not yet implemented")
    }

    override fun evaluate(testData: DataFrame<*>, target: DataFrame<*>): EvaluationMetrics {
        TODO("Not yet implemented")
    }
}
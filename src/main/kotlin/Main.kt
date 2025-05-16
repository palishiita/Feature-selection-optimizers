package com.technosudo

import algorithms.binary.TLBO
import com.technosudo.algorithms.fitness.FitnessFunction
import com.technosudo.algorithms.optimizers.GWO
import com.technosudo.algorithms.optimizers.Optimizer
import com.technosudo.data.DataLoader
import com.technosudo.data.DataProcessor.minMaxNormalize
import com.technosudo.evaluation.wrappers.RandomForestWrapper
import org.jetbrains.kotlinx.dataframe.api.toDataFrame
import smile.classification.RandomForest

fun main() {

    val dataLoaders = listOf(
        DataLoader.bcw())
    val optimizers = listOf(
        GWO(),
        TLBO()
    )

    for (dataLoader in dataLoaders) {
        for (data in dataLoader) {
            val train = data.first      //X features
            val target = data.second    //Y melignant or not

//            val trainOptimized = optimizers.first().optimize(train, fitness?)

            val model = RandomForestWrapper().fit(train, target.toDataFrame())
//            val modelOptimized = RandomForestWrapper().fit(trainOptimized, target.toDataFrame())
            model.predict(train)
            model.evaluate(train, target.toDataFrame())
        }
    }
}
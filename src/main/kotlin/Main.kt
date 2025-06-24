package com.technosudo

import com.technosudo.algorithms.fitness.FitnessFunctionImplementation
import com.technosudo.algorithms.optimizers.GWO
import com.technosudo.algorithms.optimizers.TLBO
import com.technosudo.data.DataLoader
import com.technosudo.evaluation.wrappers.RandomForestWrapper
import org.jetbrains.kotlinx.dataframe.api.select
import org.jetbrains.kotlinx.dataframe.api.take
import org.jetbrains.kotlinx.dataframe.api.toDataFrame
import kotlin.random.Random

fun main() {
    val dataLoaders = mapOf(
        "BCW" to DataLoader.bcw(),
        "Arrhythmia" to DataLoader.arrhythmia(),
        "Semi-conductor" to DataLoader.semiConductor()
    )

    val optimizers = listOf(
        GWO(name = "Binary Grey Wolf Optimizer", dataName = "name", populationSize = 10, maxIterations = 30),
        TLBO(name = "Teacher Learning Based Optimizer", dataName = "name", populationSize = 10, maxIterations = 30)
    )

    for ((name, loader) in dataLoaders) {
        println("Loading dataset: $name")

        for ((dataX, dataY) in loader) {
            println("Loaded $name")
            println("-> Feature rows: ${dataX.rowsCount()} | columns: ${dataX.columnNames().size}")
            println("-> Labels: ${dataY.size()} entries")
            println("-> Label sample: ${dataY.values().take(5)}")
            println("-> Label distribution: ${dataY.values().groupingBy { it }.eachCount()}")
            println("-> All feature columns: ${dataX.columnNames()}")
            println("-> Sample feature rows:")
            println(dataX.take(3).toString())

            val rowCount = dataX.rowsCount()
            val indexes = (0 until rowCount).shuffled(Random(42))
            val trainSize = (rowCount * 0.8).toInt()
            val trainIndexes = indexes.take(trainSize)
            val testIndexes = indexes.drop(trainSize)

            val trainX = trainIndexes.map { dataX[it] }.toDataFrame()
            val trainY = trainIndexes.map { dataY[it] }
            val testX = testIndexes.map { dataX[it] }.toDataFrame()
            val testY = testIndexes.map { dataY[it] }

            val rfBase = RandomForestWrapper()
            rfBase.fit(trainX, trainY.toDataFrame())
            val evaluationBase = rfBase.evaluate(testX, testY.toDataFrame())

            val fitness = FitnessFunctionImplementation(dataY.toDataFrame())
            for (optimizer in optimizers) {
                // Run optimizer to get best feature mask
                println("\nRunning ${optimizer.name} with ${optimizer.populationSize} wolves for ${optimizer.maxIterations} iterations...")
                val result = optimizer.optimize(dataX, fitness)

                val bestMask = result[0].values().map { (it as Number).toInt() }
                val selectedCount = bestMask.count { it == 1 }
                println("\nOptimization complete.")
                println("Selected $selectedCount / ${bestMask.size} features.")

                var selectedColumns = dataX.columnNames()
                    .filterIndexed { index, _ -> bestMask.getOrNull(index) == 1 }

                if (selectedColumns.isEmpty()) {
                    println("No features selected. Falling back to all features.")
                    selectedColumns = dataX.columnNames()
                }
                println("Selected columns: $selectedColumns")

                val selectedData = dataX.select(*selectedColumns.toTypedArray())
                val trainXoptimized = trainIndexes.map { selectedData[it] }.toDataFrame()
                val testXoptimized = testIndexes.map { selectedData[it] }.toDataFrame()

                val rfOptimized = RandomForestWrapper()
                rfOptimized.fit(trainXoptimized, trainY.toDataFrame())
                println("Model trained successfully for $name.")

                // Evaluate
                val evaluationOptimised = rfOptimized.evaluate(testXoptimized, testY.toDataFrame())
                println("\nEvaluation Metrics Base:")
                println("Accuracy: ${"%.4f".format(evaluationBase.accuracy)}")
                println("Precision: ${"%.4f".format(evaluationBase.precision)}")
                println("Recall: ${"%.4f".format(evaluationBase.recall)}")
                println("F1 Score: ${"%.4f".format(evaluationBase.f1Score)}")

                println("\nEvaluation Metrics Optimized:")
                println("Accuracy: ${"%.4f".format(evaluationOptimised.accuracy)}")
                println("Precision: ${"%.4f".format(evaluationOptimised.precision)}")
                println("Recall: ${"%.4f".format(evaluationOptimised.recall)}")
                println("F1 Score: ${"%.4f".format(evaluationOptimised.f1Score)}")

                println("\nFinal Summary for $name:")
                println("Selected Features: $selectedCount / ${dataX.columnNames().size}")
            }

        }

        println("\n" + "-".repeat(60) + "\n")
    }
}

package com.technosudo

import com.technosudo.algorithms.fitness.FitnessFunctionImplementation
import com.technosudo.algorithms.optimizers.GWO
//import com.technosudo.algorithms.optimizers.TLBO
import com.technosudo.data.DataLoader
import com.technosudo.evaluation.wrappers.RandomForestWrapper
import com.technosudo.taguchi.TaguchiExperiment
//import org.jetbrains.kotlinx.dataframe.api.columnNames
//import org.jetbrains.kotlinx.dataframe.api.select
import org.jetbrains.kotlinx.dataframe.api.take
import org.jetbrains.kotlinx.dataframe.api.toDataFrame
import kotlin.random.Random

fun main() {
    val dataLoaders = mapOf(
        "BCW" to DataLoader.bcw(),
//        "Arrhythmia" to DataLoader.arrhythmia(),
//        "Semi-conductor" to DataLoader.semiConductor()
    )

    // loading taguchi experiment
    //val taguchi = TaguchiExperiment()

    for ((name, loader) in dataLoaders) {
        println("Loading dataset: $name")

        val optimizers = listOf(
            GWO(name = "Binary Grey Wolf Optimizer", dataName = name, populationSize = 50, maxIterations = 100),
//            TLBO(name = "Teacher Learning Based Optimizer", dataName = name, populationSize = 10, maxIterations = 30)
        )

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
                // --- MODIFICATION START ---
                // The logic for handling the optimizer result has been updated.

                println("\nRunning ${optimizer.name}...")

                // 1. Run the optimizer. The result is now the final DataFrame with selected features.
                // We will call it `selectedData` to make its purpose clear.
                var selectedData = optimizer.optimize(dataX, fitness)
                println("\nOptimization complete.")

                // 2. Get the names of the selected *feature* columns for logging.
                // The optimizer returns features + the target column, so we drop the last column name
                // (which is the target) to get a list of just the selected features.
                var selectedFeatureColumns = selectedData.columnNames().dropLast(1)

                // 3. Handle the case where the optimizer selects no features.
                if (selectedFeatureColumns.isEmpty()) {
                    println("Warning: No features were selected. Falling back to using all original features.")
                    // If no features were selected, we revert to using the original full dataset.
                    selectedData = dataX
                    selectedFeatureColumns = dataX.columnNames().dropLast(1)
                }

                val selectedCount = selectedFeatureColumns.size
                // The total number of features is the column count of the original data minus the target column.
                val totalFeatureCount = dataX.columnNames().size - 1

                println("Selected $selectedCount / $totalFeatureCount features.")
                println("Selected columns: $selectedFeatureColumns")

                // 4. Create optimized train/test sets directly from `selectedData`.
                // The old, redundant step of re-selecting data is no longer needed.
                val trainXoptimized = trainIndexes.map { selectedData[it] }.toDataFrame()
                val testXoptimized = testIndexes.map { selectedData[it] }.toDataFrame()

                // --- MODIFICATION END ---

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
                println("Selected Features: $selectedCount / $totalFeatureCount")
            }
        }
        println("\n" + "-".repeat(60) + "\n")
    }
}
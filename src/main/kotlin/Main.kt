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
    val dataLoaders = listOf(
        "BCW" to DataLoader.bcw()
        // "Arrhythmia" to DataLoader.arrhythmia(),
        // "Leukemia" to DataLoader.leukemia()
    )

    for ((name, loader) in dataLoaders) {
        println("Loading dataset: $name")

        for ((features, labels) in loader) {
            println("Loaded $name")
            println("-> Feature rows: ${features.rowsCount()} | columns: ${features.columnNames().size}")
            println("-> Labels: ${labels.size()} entries")
            println("-> Label sample: ${labels.values().take(5)}")
            println("-> Label distribution: ${labels.values().groupingBy { it }.eachCount()}")
            println("-> All feature columns: ${features.columnNames()}")
            println("-> Sample feature rows:")
            println(features.take(3).toString())

            // Run optimizer to get best feature mask
            val optimizer = GWO(name = "Binary Grey Wolf Optimizer", populationSize = 10, maxIterations = 30)
            val optimizer2 = TLBO(name = "Teacher Learning Based Optimizer", populationSize = 10, maxIterations = 30)
            println("\nRunning ${optimizer.name} with ${optimizer.populationSize} wolves for ${optimizer.maxIterations} iterations...")
            println("\nRunning ${optimizer2.name} with ${optimizer2.populationSize} wolves for ${optimizer2.maxIterations} iterations...")

            val fitness = FitnessFunctionImplementation(labels.toDataFrame())
            val result = optimizer.optimize(features, fitness)

            val bestMask = result[0].values().map { (it as Number).toInt() }
            val selectedCount = bestMask.count { it == 1 }
            println("\nOptimization complete.")
            println("Selected $selectedCount / ${bestMask.size} features.")

            var selectedColumns = features.columnNames()
                .filterIndexed { index, _ -> bestMask.getOrNull(index) == 1 }

            if (selectedColumns.isEmpty()) {
                println("No features selected. Falling back to all features.")
                selectedColumns = features.columnNames()
            }

            println("Selected columns: $selectedColumns")

            val selectedData = features.select(*selectedColumns.toTypedArray())

            // Split train/test (80/20)
            val rowCount = selectedData.rowsCount()
            val indices = (0 until rowCount).shuffled(Random(42))
            val trainSize = (rowCount * 0.8).toInt()
            val trainIndices = indices.take(trainSize)
            val testIndices = indices.drop(trainSize)

            val trainFeatures = trainIndices.map { selectedData[it] }.toDataFrame()
            val testFeatures = testIndices.map { selectedData[it] }.toDataFrame()
            val trainLabels = trainIndices.map { labels[it] }
            val testLabels = testIndices.map { labels[it] }

            // Train and predict
            val rf = RandomForestWrapper()
            rf.fit(trainFeatures, trainLabels.toDataFrame())
            println("Model trained successfully for $name.")

            val predictions = rf.predict(testFeatures)
            val actual = testLabels.mapNotNull { (it as? Number)?.toDouble() }

            println("\nIndex\tPredicted\tActual")
            predictions.take(10).forEachIndexed { i, pred ->
                val act = actual.getOrNull(i)
                println("$i\t$pred\t\t$act")
            }

            // Evaluate
            val evaluation = rf.evaluate(testFeatures, testLabels.toDataFrame())
            println("\nEvaluation Metrics:")
            println("Accuracy: ${"%.4f".format(evaluation.accuracy)}")
            println("Precision: ${"%.4f".format(evaluation.precision)}")
            println("Recall: ${"%.4f".format(evaluation.recall)}")
            println("F1 Score: ${"%.4f".format(evaluation.f1Score)}")

            println("\nFinal Summary for $name:")
            println("Selected Features: $selectedCount / ${features.columnNames().size}")
        }

        println("\n" + "-".repeat(60) + "\n")
    }
}

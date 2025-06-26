package com.technosudo

import com.technosudo.algorithms.fitness.FitnessFunctionImplementation
import com.technosudo.algorithms.optimizers.GWO
import com.technosudo.algorithms.optimizers.Optimizer
import com.technosudo.algorithms.optimizers.TLBO
import com.technosudo.data.DataLoader
import com.technosudo.evaluation.wrappers.RandomForestWrapper
import com.technosudo.taguchi.TaguchiExperiment
import org.jetbrains.kotlinx.dataframe.api.*
import kotlin.random.Random

fun main() {
    val dataLoaders = mapOf(
//        "BCW" to DataLoader.bcw(),
        "Arrhythmia" to DataLoader.arrhythmia(),
//        "Semi-conductor" to DataLoader.semiConductor()
    )

    val optimizerSelected = Optimizer.TLBO

    for ((name, loader) in dataLoaders) {
        println("Loading dataset: $name")

        for ((dataX, dataY) in loader) {
            println("-> Feature rows: ${dataX.rowsCount()} | columns: ${dataX.columnNames().size}")

            val rowCount = dataX.rowsCount()
            val indexes = (0 until rowCount).shuffled(Random(42))
            val trainSize = (rowCount * 0.8).toInt()
            val trainIndexes = indexes.take(trainSize)
            val testIndexes = indexes.drop(trainSize)

            val trainX = trainIndexes.map { dataX[it] }.toDataFrame()
            val trainY = trainIndexes.map { dataY[it] }
            val testX = testIndexes.map { dataX[it] }.toDataFrame()
            val testY = testIndexes.map { dataY[it] }

            val taguchi = TaguchiExperiment()

            val rfBase = RandomForestWrapper()
            rfBase.fit(trainX, trainY.toDataFrame())
            val baselineEval = rfBase.evaluate(testX, testY.toDataFrame())
            taguchi.recordBaseline(name, baselineEval.accuracy)

            println("\nBaseline: Acc=${"%.4f".format(baselineEval.accuracy)}, " +
                    "Prec=${"%.4f".format(baselineEval.precision)}, " +
                    "Rec=${"%.4f".format(baselineEval.recall)}, " +
                    "F1=${"%.4f".format(baselineEval.f1Score)}")

            val fitnessFunction = FitnessFunctionImplementation(dataY.toDataFrame())
            val configurations = taguchi.generateConfigurations()

            for (config in configurations) {
                val popSize = config.parameters["populationSize"] as Int
                val mutationRate = config.parameters["mutationRate"] as Double

                println("\nRunning Config ${config.experimentId} → populationSize=$popSize, mutationRate=$mutationRate")

                val optimizer = when (optimizerSelected) {
                    Optimizer.GWO -> GWO(
                        name = "GWO-Taguchi",
                        dataName = name,
                        populationSize = popSize,
                        maxIterations = 150,
                        mutationRate = mutationRate,
                        maxSolutions = 1500
                    )
                    Optimizer.TLBO -> TLBO(
                        name = "TLBO-Taguchi",
                        dataName = name,
                        populationSize = popSize,
                        maxIterations = 150,
                        mutationRate = mutationRate,
                        maxSolutions = 1500
                    )
                    else -> throw Exception("Unknown optimizer selected")
                }

                val startTime = System.currentTimeMillis()
                var selectedData = optimizer.optimize(dataX, fitnessFunction)
                val endTime = System.currentTimeMillis()
                val runtime = endTime - startTime

                var selectedFeatureColumns = selectedData.columnNames().dropLast(1)
                if (selectedFeatureColumns.isEmpty()) {
                    selectedData = dataX
                    selectedFeatureColumns = dataX.columnNames().dropLast(1)
                }

                val trainXopt = trainIndexes.map { selectedData[it] }.toDataFrame()
                val testXopt = testIndexes.map { selectedData[it] }.toDataFrame()

                val rfOpt = RandomForestWrapper()
                rfOpt.fit(trainXopt, trainY.toDataFrame())
                val optEval = rfOpt.evaluate(testXopt, testY.toDataFrame())

                taguchi.recordResult(
                    config = config,
                    fitness = optEval.accuracy,
                    accuracy = optEval.accuracy,
                    precision = optEval.precision,
                    recall = optEval.recall,
                    f1Score = optEval.f1Score,
                    featuresSelected = selectedFeatureColumns.size,
                    totalFeatures = dataX.columnNames().size - 1,
                    baselineAccuracy = baselineEval.accuracy,
                    selectedFeatureMask = selectedFeatureColumns.map { dataX.columnNames().indexOf(it) },
                    runtime = runtime
                )

                println("→ Config ${config.experimentId} | Acc=${"%.4f".format(optEval.accuracy)}, " +
                        "Prec=${"%.4f".format(optEval.precision)}, " +
                        "Rec=${"%.4f".format(optEval.recall)}, F1=${"%.4f".format(optEval.f1Score)}")
            }

            val optimalConfig = taguchi.analyzeAndFindOptimal()
            println("\nOptimal configuration: $optimalConfig")

            taguchi.exportToCSV()
            println(taguchi.getSummary())
        }
        println("\n" + "=".repeat(60) + "\n")
    }
}

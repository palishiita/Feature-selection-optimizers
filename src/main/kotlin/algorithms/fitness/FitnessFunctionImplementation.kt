package com.technosudo.algorithms.fitness

import com.technosudo.evaluation.wrappers.RandomForestWrapper
import com.technosudo.evaluation.EvaluationMetrics
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.*
import kotlin.random.Random

class FitnessFunctionImplementation(
    private val target: DataFrame<*>
) : FitnessFunction {

    override fun evaluate(dataset: DataFrame<*>, featureMask: List<Int>): Double {
        val selectedColumns = dataset.columnNames()
            .filterIndexed { index, _ -> featureMask.getOrNull(index) == 1 }

        val minRequiredFeatures = (dataset.columnNames().size * 0.1).toInt().coerceAtLeast(1) // Require at least 10%

        // Penalize zero or too few features
        if (selectedColumns.size < minRequiredFeatures) {
            return 0.0
        }

        val filteredData = dataset.select(*selectedColumns.toTypedArray())
        val rowCount = filteredData.rowsCount()
        val indices = (0 until rowCount).shuffled(Random(42))
        val trainSize = (rowCount * 0.8).toInt()

        val trainIndices = indices.take(trainSize)
        val testIndices = indices.drop(trainSize)

        val trainFeatures = trainIndices.map { filteredData[it] }.toDataFrame()
        val testFeatures = testIndices.map { filteredData[it] }.toDataFrame()
        val trainLabels = trainIndices.map { target[it] }.toDataFrame()
        val testLabels = testIndices.map { target[it] }.toDataFrame()

        return try {
            val rf = RandomForestWrapper()
            rf.fit(trainFeatures, trainLabels)
            val metrics = rf.evaluate(testFeatures, testLabels)

            // Bonus: Slight penalty for using too many features
            val usagePenalty = selectedColumns.size.toDouble() / dataset.columnNames().size
            metrics.accuracy * (1.0 - 0.1 * usagePenalty)  // Penalize large feature sets slightly
        } catch (e: Exception) {
            0.0
        }
    }

    override fun evaluateDetailed(dataset: DataFrame<*>, featureMask: List<Int>): FitnessResult {
        val selectedColumns = dataset.columnNames()
            .filterIndexed { index, _ -> featureMask.getOrNull(index) == 1 }

        val minRequiredFeatures = (dataset.columnNames().size * 0.1).toInt().coerceAtLeast(1)
        if (selectedColumns.size < minRequiredFeatures) {
            return FitnessResult(0.0, EvaluationMetrics(0.0, 0.0, 0.0, 0.0))
        }

        val filteredData = dataset.select(*selectedColumns.toTypedArray())
        val rowCount = filteredData.rowsCount()
        val indices = (0 until rowCount).shuffled(Random(42))
        val trainSize = (rowCount * 0.8).toInt()

        val trainIndices = indices.take(trainSize)
        val testIndices = indices.drop(trainSize)

        val trainFeatures = trainIndices.map { filteredData[it] }.toDataFrame()
        val testFeatures = testIndices.map { filteredData[it] }.toDataFrame()
        val trainLabels = trainIndices.map { target[it] }.toDataFrame()
        val testLabels = testIndices.map { target[it] }.toDataFrame()

        return try {
            val rf = RandomForestWrapper()
            rf.fit(trainFeatures, trainLabels)
            val metrics = rf.evaluate(testFeatures, testLabels)

            val usagePenalty = selectedColumns.size.toDouble() / dataset.columnNames().size
            val fitness = metrics.accuracy * (1.0 - 0.1 * usagePenalty)
            FitnessResult(fitness, metrics)
        } catch (e: Exception) {
            FitnessResult(0.0, EvaluationMetrics(0.0, 0.0, 0.0, 0.0))
        }
    }

}

package com.technosudo.evaluation.wrappers

import com.technosudo.evaluation.ClassifierWrapper
import com.technosudo.evaluation.EvaluationMetrics
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.rows
import org.jetbrains.kotlinx.dataframe.api.toDataFrame
import org.jetbrains.kotlinx.dataframe.api.values
import smile.classification.RandomForest
import smile.data.formula.Formula
import smile.data.DataFrame as SmileFrame
import smile.data.vector.IntVector

class RandomForestWrapper(private val batchSize: Int = 1000) : ClassifierWrapper {

    private lateinit var model: RandomForest
    private lateinit var labelEncoder: Map<Double, Int>
    private lateinit var labelDecoder: Map<Int, Double>

    // Convert Kotlin DataFrame to Smile DataFrame
    private fun smileDataFrame(df: DataFrame<*>): SmileFrame {
        val columnNames = df.columnNames()
        val rowCount = df.rowsCount()

        // Ensure all columns have consistent size
        df.columns().forEach { column ->
            if (column.size() != rowCount) {
                throw IllegalArgumentException("Column '${column.name()}' has inconsistent size. Expected $rowCount but found ${column.size()}.")
            }
        }

        // Convert DataFrame into a transposed 2D array with rows of Doubles
        val data: Array<DoubleArray> = Array(rowCount) { DoubleArray(columnNames.size) }

        for (chunk in (0 until rowCount).chunked(batchSize)) {
            chunk.forEach { rowIndex ->
                val row = DoubleArray(columnNames.size) { colIndex ->
                    val value = df[rowIndex][colIndex]
                    when (value) {
                        is Number -> value.toDouble()
                        is String -> value.toDoubleOrNull() ?: Double.NaN
                        else -> Double.NaN
                    }
                }
                data[rowIndex] = row
            }
        }

        // Impute NaNs with column-wise means
        val colMeans = DoubleArray(columnNames.size) { col ->
            val values = data.map { it[col] }.filter { !it.isNaN() }
            if (values.isEmpty()) 0.0 else values.average()
        }
        val imputedData = data.map { row ->
            DoubleArray(row.size) { i -> if (row[i].isNaN()) colMeans[i] else row[i] }
        }.toTypedArray()

        return SmileFrame.of(imputedData, *columnNames.toTypedArray())
    }


    private fun mergeFeaturesAndLabels(
        features: SmileFrame,
        labels: List<Int>
    ): SmileFrame {
        val labelVector = IntVector("class", labels.toIntArray())
        val labelFrame = SmileFrame(labelVector)
        return features.merge(labelFrame)
    }


    override fun fit(train: DataFrame<*>, target: DataFrame<*>): ClassifierWrapper {
        // Convert features (Kotlin DataFrame → Smile DataFrame)
        val x = smileDataFrame(train)

        // Encode labels (e.g., 0.0, 1.0, etc. → Ints)
        val rawLabels = target.values().map { (it as Number).toDouble() }
        labelEncoder = rawLabels.distinct().sorted().withIndex().associate { it.value to it.index }
        labelDecoder = labelEncoder.entries.associate { (k, v) -> v to k }
        val encodedLabels = rawLabels.map { labelEncoder[it] ?: error("Unknown label: $it") }.toList()

        // Convert label list to Smile IntVector → wrap in Smile DataFrame
        val labelVector = IntVector("class", encodedLabels.toIntArray())
        val dataWithLabels = x.merge(smile.data.DataFrame(labelVector))  // merged dataset

        // Train RandomForest using Smile’s formula-based API
        model = RandomForest.fit(Formula.lhs("class"), dataWithLabels)

        return this
    }


    override fun predict(test: DataFrame<*>): List<Double> {
        val rowCount = test.rowsCount()
        val predictions = mutableListOf<Int>()
        for (chunk in (0 until rowCount).chunked(batchSize)) {
            val batch = chunk.map { test[it] }.toDataFrame()
            val smileBatch = smileDataFrame(batch)
            predictions += model.predict(smileBatch).toList()
        }

        // Decode Int labels back to original Double labels
        return predictions.map { labelDecoder[it] ?: error("Unknown predicted label: $it") }
    }


    override fun evaluate(test: DataFrame<*>, target: DataFrame<*>): EvaluationMetrics {
        val predictions = predict(test)
        val trueLabels = target.rows().map { row ->
            val value = row.values().firstOrNull()
            (value as? Number)?.toDouble() ?: Double.NaN
        }

        val uniqueLabels = (trueLabels + predictions).filterNot { it.isNaN() }.distinct().sorted()

        var tp = 0
        val total = predictions.size

        val labelWiseStats = uniqueLabels.associateWith { label ->
            var tpLocal = 0
            var fp = 0
            var fn = 0

            for (i in predictions.indices) {
                val pred = predictions[i]
                val actual = trueLabels.getOrNull(i)

                if (pred == label && actual == label) tpLocal++
                if (pred == label && actual != label) fp++
                if (pred != label && actual == label) fn++
            }

            tp += tpLocal

            val precision = if (tpLocal + fp > 0) tpLocal.toDouble() / (tpLocal + fp) else 0.0
            val recall = if (tpLocal + fn > 0) tpLocal.toDouble() / (tpLocal + fn) else 0.0
            val f1 = if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0.0

            Triple(precision, recall, f1)
        }

        val avgPrecision = labelWiseStats.values.map { it.first }.average()
        val avgRecall = labelWiseStats.values.map { it.second }.average()
        val avgF1 = labelWiseStats.values.map { it.third }.average()
        val accuracy = tp.toDouble() / total

        return EvaluationMetrics(
            accuracy = accuracy,
            precision = avgPrecision,
            recall = avgRecall,
            f1Score = avgF1,
            //confusionMatrix = TODO()
        )
    }


}

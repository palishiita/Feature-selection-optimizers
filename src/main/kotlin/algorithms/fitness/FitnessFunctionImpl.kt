package com.technosudo.algorithms.fitness

import com.technosudo.evaluation.ClassifierWrapper
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.select

class FitnessFunctionImpl(
    private val classifier: ClassifierWrapper,
    private val alpha: Double = 0.9,
    private val beta: Double = 0.1
) : FitnessFunction {

    override fun evaluate(
        dataset: DataFrame<*>,
        featureMask: List<Int>
    ): Double {
        val totalFeatures = featureMask.size
        val selectedIndices = featureMask.withIndex().filter { it.value == 1 }.map { it.index }
        if (selectedIndices.isEmpty()) return Double.MAX_VALUE // Penalize empty selection

        // Select columns based on mask
        val featureCols = dataset.columns()
        val selectedCols = selectedIndices.map { featureCols[it] }
        val selectedDf = dataset.select { selectedCols }

        // You need to provide the target column separately for the classifier
        // This assumes the last column is the target; adjust as needed
        val targetCol = dataset.columns().last()
        val targetDf = dataset.select { targetCol }

        // Evaluate error rate using the classifier
        val model = classifier.fit(selectedDf, targetDf)
        val metrics = model.evaluate(selectedDf, targetDf)
        val errorRate = 1.0 - metrics.accuracy

        // Fitness function: α * error + β * (#selected / total)
        val featurePenalty = selectedIndices.size.toDouble() / totalFeatures
        return alpha * errorRate + beta * featurePenalty
    }
}
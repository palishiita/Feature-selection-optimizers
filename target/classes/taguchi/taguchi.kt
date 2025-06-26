package com.technosudo.taguchi

import com.technosudo.algorithms.fitness.FitnessFunctionImplementation
import com.technosudo.algorithms.optimizers.GWO
import com.technosudo.data.DataLoader
import org.jetbrains.kotlinx.dataframe.api.toDataFrame

object TaguchiExperiment {
    // Define Taguchi L9 array for 3 factors, 3 levels each
    private val taguchiL9 = listOf(
        Triple(10, 0.01, 30),
        Triple(10, 0.02, 50),
        Triple(10, 0.05, 100),
        Triple(30, 0.01, 50),
        Triple(30, 0.02, 100),
        Triple(30, 0.05, 30),
        Triple(50, 0.01, 100),
        Triple(50, 0.02, 30),
        Triple(50, 0.05, 50)
    )

    fun run() {
        val dataLoaders = mapOf(
            "Arrhythmia" to DataLoader.arrhythmia()
        )

        for ((name, loader) in dataLoaders) {
            for ((features, labels) in loader) {
                val fitness = FitnessFunctionImplementation(labels.toDataFrame())
                println("Taguchi experiment for $name")

                for ((expIdx, params) in taguchiL9.withIndex()) {
                    val (populationSize, mutationRate, maxIterations) = params
                    val optimizer = GWO(
                        name = "GWO_Taguchi",
                        dataName = name,
                        populationSize = populationSize,
                        mutationRate = mutationRate,
                        maxIterations = maxIterations
                    )
                    println("Experiment ${expIdx + 1}: pop=$populationSize, mut=$mutationRate, iter=$maxIterations")
                    val result = optimizer.optimize(features, fitness)
                    val bestMask = result[0].values().map { (it as Number).toInt() }
                    val selectedColumns = features.columnNames()
                        .filterIndexed { index, _ -> bestMask.getOrNull(index) == 1 }
                    println("Selected columns: $selectedColumns")
                }
            }
        }
    }
}
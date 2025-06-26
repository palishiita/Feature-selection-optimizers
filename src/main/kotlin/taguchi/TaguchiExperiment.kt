package com.technosudo.taguchi

import java.io.File
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import kotlin.math.log10

data class TaguchiParameter(
    val name: String,
    val levels: List<Any>
)

data class ExperimentConfiguration(
    val experimentId: Int,
    val parameters: Map<String, Any>
)

data class ExperimentResult(
    val configuration: ExperimentConfiguration,
    val fitness: Double,
    val accuracy: Double,
    val precision: Double,
    val recall: Double,
    val f1Score: Double,
    val featuresSelected: Int,
    val totalFeatures: Int,
    val baselineAccuracy: Double,
    val selectedFeatureMask: List<Int>,
    val runtime: Long
) {
    val featureReduction: Double
        get() = ((totalFeatures - featuresSelected).toDouble() / totalFeatures) * 100

    val accuracyImprovement: Double
        get() = accuracy - baselineAccuracy

    val featureEfficiency: Double
        get() = if (featuresSelected > 0) accuracy / featuresSelected else 0.0
}

data class SNRatioResult(
    val parameter: String,
    val level: Any,
    val snRatio: Double,
    val meanResponse: Double,
    val count: Int
)

class TaguchiExperiment {

    companion object {
        private val L9_ARRAY = arrayOf(
            intArrayOf(1, 1), intArrayOf(1, 2), intArrayOf(1, 3),
            intArrayOf(2, 1), intArrayOf(2, 2), intArrayOf(2, 3),
            intArrayOf(3, 1), intArrayOf(3, 2), intArrayOf(3, 3)
        )
    }

    private val parameters = listOf(
        TaguchiParameter("populationSize", listOf(50, 75, 100)),
        TaguchiParameter("mutationRate", listOf(0.03, 0.05, 0.07))
    )

    private val results = mutableListOf<ExperimentResult>()
    private val baselineResults = mutableMapOf<String, Double>()

    fun recordBaseline(datasetName: String, baselineAccuracy: Double) {
        baselineResults[datasetName] = baselineAccuracy
    }

    fun getBaseline(datasetName: String): Double {
        return baselineResults[datasetName] ?: 0.0
    }

    fun generateConfigurations(): List<ExperimentConfiguration> {
        val configurations = mutableListOf<ExperimentConfiguration>()
        L9_ARRAY.forEachIndexed { index, row ->
            val paramMap = mutableMapOf<String, Any>()
            row.forEachIndexed { i, level ->
                val param = parameters[i]
                paramMap[param.name] = param.levels[level - 1]
            }
            configurations.add(ExperimentConfiguration(index + 1, paramMap))
        }
        return configurations
    }

    fun recordResult(
        config: ExperimentConfiguration,
        fitness: Double,
        accuracy: Double,
        precision: Double,
        recall: Double,
        f1Score: Double,
        featuresSelected: Int,
        totalFeatures: Int,
        baselineAccuracy: Double,
        selectedFeatureMask: List<Int>,
        runtime: Long
    ) {
        results.add(
            ExperimentResult(
                configuration = config,
                fitness = fitness,
                accuracy = accuracy,
                precision = precision,
                recall = recall,
                f1Score = f1Score,
                featuresSelected = featuresSelected,
                totalFeatures = totalFeatures,
                baselineAccuracy = baselineAccuracy,
                selectedFeatureMask = selectedFeatureMask,
                runtime = runtime
            )
        )
    }

    fun recordResult(config: ExperimentConfiguration, fitness: Double, runtime: Long) {
        recordResult(config, fitness, fitness, 0.0, 0.0, 0.0, 0, 1, 0.0, emptyList(), runtime)
    }

    private fun calculateSNRatio(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val sumOfInverseSquares = values.sumOf { if (it > 0) 1.0 / (it * it) else 1.0 }
        return -10 * log10(sumOfInverseSquares / values.size)
    }

    fun analyzeAndFindOptimal(): Map<String, Any> {
        val analysisResults = mutableMapOf<String, List<SNRatioResult>>()
        parameters.forEach { param ->
            val snResults = param.levels.map { level ->
                val matching = results.filter {
                    it.configuration.parameters[param.name] == level
                }
                val fitnessValues = matching.map { it.fitness }
                SNRatioResult(
                    parameter = param.name,
                    level = level,
                    snRatio = calculateSNRatio(fitnessValues),
                    meanResponse = fitnessValues.average(),
                    count = fitnessValues.size
                )
            }.sortedByDescending { it.snRatio }
            analysisResults[param.name] = snResults
        }

        generateReport(analysisResults)
        return analysisResults.mapValues { (_, list) -> list.first().level }
    }

    private fun generateReport(analysisResults: Map<String, List<SNRatioResult>>) {
        val timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss"))
        val file = File("taguchi_report_$timestamp.txt")

        file.printWriter().use { out ->
            out.println("TAGUCHI ANALYSIS REPORT")
            out.println("=".repeat(60))
            out.println("Generated: ${LocalDateTime.now()}")
            out.println("Total Experiments: ${results.size}")
            out.println()

            out.println("PARAMETER ANALYSIS (S/N Ratio):")
            out.println("-".repeat(60))
            analysisResults.forEach { (param, snList) ->
                out.println("\nParameter: $param")
                out.println("Level\tS/N Ratio\tMean Fitness\tCount")
                snList.forEach {
                    out.println("${it.level}\t${"%.3f".format(it.snRatio)}\t\t${"%.3f".format(it.meanResponse)}\t\t${it.count}")
                }
                val best = snList.maxByOrNull { it.snRatio }
                out.println("★ Best: ${best?.level} (S/N: ${"%.3f".format(best?.snRatio ?: 0.0)})")
            }

            out.println("\n" + "=".repeat(60))
            out.println("DETAILED RESULTS:")
            out.println("Config\tFitness\tAccuracy\tPrecision\tRecall\tF1\tSelected\tReduction%\tImprovement\tEfficiency\tRuntime(ms)")
            results.forEach {
                out.println("${it.configuration.experimentId}\t" +
                        "${"%.4f".format(it.fitness)}\t" +
                        "${"%.4f".format(it.accuracy)}\t" +
                        "${"%.4f".format(it.precision)}\t" +
                        "${"%.4f".format(it.recall)}\t" +
                        "${"%.4f".format(it.f1Score)}\t" +
                        "${it.featuresSelected}/${it.totalFeatures}\t" +
                        "${"%.1f".format(it.featureReduction)}\t" +
                        "${"%.4f".format(it.accuracyImprovement)}\t" +
                        "${"%.6f".format(it.featureEfficiency)}\t" +
                        "${it.runtime}")
            }

            out.println("\n" + "=".repeat(60))
            out.println("SUMMARY:")
            if (results.isNotEmpty()) {
                val acc = results.map { it.accuracy }
                out.println("Best Accuracy: ${"%.4f".format(acc.maxOrNull() ?: 0.0)}")
                out.println("Average Accuracy: ${"%.4f".format(acc.average())}")
            }
        }

        println("Report saved to: ${file.absolutePath}")
    }

    fun exportToCSV() {
        val timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss"))
        val csvFile = File("taguchi_results_$timestamp.csv")

        val fixedMaxIter = 150
        val fixedMaxSolutions = 1500

        csvFile.printWriter().use { out ->
            out.println("experiment_id,population_size,mutation_rate,max_iterations,max_solutions," +
                    "fitness,accuracy,precision,recall,f1_score,features_selected,total_features,feature_reduction_percent," +
                    "baseline_accuracy,accuracy_improvement,feature_efficiency,runtime_ms,selected_features")

            results.forEach { r ->
                val config = r.configuration
                val selectedMask = r.selectedFeatureMask.joinToString(";")
                out.println("${config.experimentId}," +
                        "${config.parameters["populationSize"]}," +
                        "${config.parameters["mutationRate"]}," +
                        "$fixedMaxIter,$fixedMaxSolutions," +
                        "${r.fitness}," +
                        "${r.accuracy}," +
                        "${r.precision}," +
                        "${r.recall}," +
                        "${r.f1Score}," +
                        "${r.featuresSelected}," +
                        "${r.totalFeatures}," +
                        "${"%.2f".format(r.featureReduction)}," +
                        "${r.baselineAccuracy}," +
                        "${"%.4f".format(r.accuracyImprovement)}," +
                        "${"%.6f".format(r.featureEfficiency)}," +
                        "${r.runtime}," +
                        "\"$selectedMask\"")
            }
        }

        println("CSV exported to: ${csvFile.absolutePath}")
    }

    fun getSummary(): String {
        val avgFeatures = results.filter { it.featuresSelected > 0 }.map { it.featuresSelected }.average()
        val avgReduction = results.filter { it.featuresSelected > 0 }.map { it.featureReduction }.average()
        return "Taguchi L9 (2x3) | Experiments: ${results.size} | Avg Features: ${"%.1f".format(avgFeatures)} | Avg Reduction: ${"%.1f".format(avgReduction)}%"
    }
}
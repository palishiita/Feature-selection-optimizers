package com.technosudo.taguchi

import java.io.File
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import kotlin.math.log10

/**
 * Data class representing a parameter to be optimized in Taguchi experiments
 */
data class TaguchiParameter(
    val name: String,
    val levels: List<Any>
)

/**
 * Data class representing a single experiment configuration
 */
data class ExperimentConfiguration(
    val experimentId: Int,
    val parameters: Map<String, Any>
)

/**
 * Data class for storing experiment results
 */
data class ExperimentResult(
    val configuration: ExperimentConfiguration,
    val fitness: Double,
    val accuracy: Double,
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
        get() = accuracy / featuresSelected // Accuracy per feature
}

/**
 * Data class for S/N ratio analysis results
 */
data class SNRatioResult(
    val parameter: String,
    val level: Any,
    val snRatio: Double,
    val meanResponse: Double,
    val count: Int
)

/**
 * Lean Taguchi experiment class that focuses only on parameter optimization logic
 * Designed to integrate with existing Main.kt infrastructure
 */
class TaguchiExperiment {
    
    companion object {
        /**
         * L9 Orthogonal Array - supports 4 factors with 3 levels each
         */
        private val L9_ARRAY = arrayOf(
            intArrayOf(1, 1, 1, 1),
            intArrayOf(1, 2, 2, 2),
            intArrayOf(1, 3, 3, 3),
            intArrayOf(2, 1, 2, 3),
            intArrayOf(2, 2, 3, 1),
            intArrayOf(2, 3, 1, 2),
            intArrayOf(3, 1, 3, 2),
            intArrayOf(3, 2, 1, 3),
            intArrayOf(3, 3, 2, 1)
        )
    }
    
    private val parameters = listOf(
        TaguchiParameter("populationSize", listOf(30, 50, 100)),
        TaguchiParameter("maxIterations", listOf(50, 100, 150)),
        TaguchiParameter("mutationRate", listOf(0.01, 0.015, 0.02)),
        TaguchiParameter("maxSolutions", listOf(500, 1000, 1500))
    )
    
    private val results = mutableListOf<ExperimentResult>()
    private val baselineResults = mutableMapOf<String, Double>() // dataset -> baseline accuracy
    
    /**
     * Record baseline performance (using all features)
     */
    fun recordBaseline(datasetName: String, baselineAccuracy: Double) {
        baselineResults[datasetName] = baselineAccuracy
    }
    
    /**
     * Get baseline accuracy for a dataset
     */
    fun getBaseline(datasetName: String): Double {
        return baselineResults[datasetName] ?: 0.0
    }
    
    /**
     * Generate L9 parameter configurations
     */
    fun generateConfigurations(): List<ExperimentConfiguration> {
        val configurations = mutableListOf<ExperimentConfiguration>()
        
        L9_ARRAY.forEachIndexed { index, row ->
            val parameterMap = mutableMapOf<String, Any>()
            
            row.forEachIndexed { paramIndex, level ->
                if (paramIndex < parameters.size) {
                    val parameter = parameters[paramIndex]
                    val value = parameter.levels[level - 1] // Convert to 0-based index
                    parameterMap[parameter.name] = value
                }
            }
            
            configurations.add(ExperimentConfiguration(index + 1, parameterMap))
        }
        
        return configurations
    }
    
    /**
     * Record experiment result with feature selection metrics
     */
    fun recordResult(
        config: ExperimentConfiguration, 
        fitness: Double, 
        accuracy: Double,
        featuresSelected: Int,
        totalFeatures: Int,
        baselineAccuracy: Double,
        selectedFeatureMask: List<Int>,
        runtime: Long
    ) {
        results.add(ExperimentResult(
            configuration = config,
            fitness = fitness,
            accuracy = accuracy,
            featuresSelected = featuresSelected,
            totalFeatures = totalFeatures,
            baselineAccuracy = baselineAccuracy,
            selectedFeatureMask = selectedFeatureMask,
            runtime = runtime
        ))
    }
    
    /**
     * Simplified record method for backward compatibility
     */
    fun recordResult(config: ExperimentConfiguration, fitness: Double, runtime: Long) {
        // Default values when feature metrics are not available
        recordResult(config, fitness, fitness, 0, 1, 0.0, emptyList(), runtime)
    }
    
    /**
     * Calculate Signal-to-Noise ratio (larger-is-better)
     */
    private fun calculateSNRatio(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val sumOfInverseSquares = values.sumOf { value ->
            if (value > 0) 1.0 / (value * value) else 1.0
        }
        return -10 * log10(sumOfInverseSquares / values.size)
    }
    
    /**
     * Analyze results and find optimal configuration
     */
    fun analyzeAndFindOptimal(): Map<String, Any> {
        val analysisResults = mutableMapOf<String, List<SNRatioResult>>()
        
        parameters.forEach { parameter ->
            val snResults = mutableListOf<SNRatioResult>()
            
            parameter.levels.forEach { level ->
                val relevantResults = results.filter { 
                    it.configuration.parameters[parameter.name] == level 
                }
                
                if (relevantResults.isNotEmpty()) {
                    val fitnessValues = relevantResults.map { it.fitness }
                    val snRatio = calculateSNRatio(fitnessValues)
                    val meanResponse = fitnessValues.average()
                    
                    snResults.add(SNRatioResult(
                        parameter = parameter.name,
                        level = level,
                        snRatio = snRatio,
                        meanResponse = meanResponse,
                        count = relevantResults.size
                    ))
                }
            }
            
            analysisResults[parameter.name] = snResults.sortedByDescending { it.snRatio }
        }
        
        // Generate report
        generateReport(analysisResults)
        
        // Find optimal configuration
        val optimalParams = mutableMapOf<String, Any>()
        analysisResults.forEach { (paramName, snResults) ->
            val bestLevel = snResults.maxByOrNull { it.snRatio }
            if (bestLevel != null) {
                optimalParams[paramName] = bestLevel.level
            }
        }
        
        return optimalParams
    }
    
    /**
     * Generate analysis report with feature selection metrics
     */
    private fun generateReport(analysisResults: Map<String, List<SNRatioResult>>) {
        val timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss"))
        val reportFile = File("taguchi_feature_selection_analysis_$timestamp.txt")
        
        reportFile.printWriter().use { out ->
            out.println("TAGUCHI METHOD FEATURE SELECTION ANALYSIS")
            out.println("\n" + "=".repeat(60))
            out.println("Generated: ${LocalDateTime.now()}")
            out.println("L9 Array: 4 factors, 3 levels, 9 experiments")
            out.println("Total Results: ${results.size}")
            out.println()
            
            // Parameter Analysis
            out.println("PARAMETER ANALYSIS (S/N Ratio - Larger is Better):")
            out.println("\n" + "=".repeat(60))
            analysisResults.forEach { (paramName, snResults) ->
                out.println("\n$paramName:")
                out.println("Level\tS/N Ratio\tMean Fitness\tCount")
                out.println("\n" + "=".repeat(40))
                snResults.forEach { result ->
                    out.println("${result.level}\t${String.format("%.3f", result.snRatio)}\t\t${String.format("%.3f", result.meanResponse)}\t\t${result.count}")
                }
                val best = snResults.maxByOrNull { it.snRatio }
                out.println("★ Best: ${best?.level} (S/N: ${String.format("%.3f", best?.snRatio ?: 0.0)})")
                out.println()
            }
            
            // Optimal Configuration
            out.println("\n" + "=".repeat(60))
            out.println("OPTIMAL CONFIGURATION:")
            out.println("\n" + "=".repeat(60))
            analysisResults.forEach { (paramName, snResults) ->
                val bestLevel = snResults.maxByOrNull { it.snRatio }
                if (bestLevel != null) {
                    out.println("$paramName: ${bestLevel.level}")
                }
            }
            
            // Feature Selection Analysis
            if (results.any { it.featuresSelected > 0 }) {
                out.println("\n" + "=".repeat(60))
                out.println("FEATURE SELECTION ANALYSIS:")
                out.println("\n" + "=".repeat(60))
                
                val avgFeaturesSelected = results.filter { it.featuresSelected > 0 }.map { it.featuresSelected }.average()
                val avgFeatureReduction = results.filter { it.featuresSelected > 0 }.map { it.featureReduction }.average()
                val avgAccuracyImprovement = results.filter { it.baselineAccuracy > 0 }.map { it.accuracyImprovement }.average()
                val bestFeatureEfficiency = results.filter { it.featuresSelected > 0 }.maxByOrNull { it.featureEfficiency }
                
                out.println("Average Features Selected: ${String.format("%.1f", avgFeaturesSelected)}")
                out.println("Average Feature Reduction: ${String.format("%.1f", avgFeatureReduction)}%")
                out.println("Average Accuracy Improvement: ${String.format("%.4f", avgAccuracyImprovement)}")
                out.println("Best Feature Efficiency: ${String.format("%.6f", bestFeatureEfficiency?.featureEfficiency ?: 0.0)} (Config ${bestFeatureEfficiency?.configuration?.experimentId})")
                
                // Best vs Worst Feature Selection
                val bestAccuracy = results.filter { it.accuracy > 0 }.maxByOrNull { it.accuracy }
                val mostEfficient = results.filter { it.featuresSelected > 0 }.minByOrNull { it.featuresSelected }
                
                if (bestAccuracy != null) {
                    out.println("\nBest Accuracy Result:")
                    out.println("  Config: ${bestAccuracy.configuration.experimentId}")
                    out.println("  Accuracy: ${String.format("%.4f", bestAccuracy.accuracy)}")
                    out.println("  Features: ${bestAccuracy.featuresSelected}/${bestAccuracy.totalFeatures}")
                    out.println("  Reduction: ${String.format("%.1f", bestAccuracy.featureReduction)}%")
                    out.println("  Improvement: ${String.format("%.4f", bestAccuracy.accuracyImprovement)}")
                }
                
                if (mostEfficient != null) {
                    out.println("\nMost Feature-Efficient Result:")
                    out.println("  Config: ${mostEfficient.configuration.experimentId}")
                    out.println("  Features: ${mostEfficient.featuresSelected}/${mostEfficient.totalFeatures}")
                    out.println("  Accuracy: ${String.format("%.4f", mostEfficient.accuracy)}")
                    out.println("  Efficiency: ${String.format("%.6f", mostEfficient.featureEfficiency)}")
                }
            }
            
            // Baseline Comparison
            if (baselineResults.isNotEmpty()) {
                out.println("\n" + "=".repeat(60))
                out.println("BASELINE vs OPTIMIZED COMPARISON:")
                out.println("\n" + "=".repeat(60))
                baselineResults.forEach { (dataset, baseline) ->
                    val datasetResults = results.filter { it.baselineAccuracy == baseline }
                    if (datasetResults.isNotEmpty()) {
                        val avgOptimized = datasetResults.map { it.accuracy }.average()
                        val avgFeatures = datasetResults.map { it.featuresSelected }.average()
                        val avgTotal = datasetResults.map { it.totalFeatures }.average()
                        
                        out.println("\nDataset: $dataset")
                        out.println("  Baseline Accuracy: ${String.format("%.4f", baseline)}")
                        out.println("  Optimized Accuracy: ${String.format("%.4f", avgOptimized)}")
                        out.println("  Improvement: ${String.format("%.4f", avgOptimized - baseline)}")
                        out.println("  Avg Features Used: ${String.format("%.1f", avgFeatures)}/${String.format("%.0f", avgTotal)}")
                        out.println("  Feature Reduction: ${String.format("%.1f", ((avgTotal - avgFeatures) / avgTotal) * 100)}%")
                    }
                }
            }
            
            // Detailed Results Table
            out.println("\n" + "=".repeat(60))
            out.println("DETAILED RESULTS:")
            out.println("\n" + "=".repeat(60))
            out.println("Config\tFitness\tAccuracy\tFeatures\tReduction%\tImprovement\tEfficiency\tRuntime(ms)")
            out.println("\n" + "=".repeat(90))
            results.forEach { result ->
                out.println("${result.configuration.experimentId}\t" +
                          "${String.format("%.4f", result.fitness)}\t" +
                          "${String.format("%.4f", result.accuracy)}\t" +
                          "${result.featuresSelected}/${result.totalFeatures}\t" +
                          "${String.format("%.1f", result.featureReduction)}\t\t" +
                          "${String.format("%.4f", result.accuracyImprovement)}\t\t" +
                          "${String.format("%.6f", result.featureEfficiency)}\t" +
                          "${result.runtime}")
            }
            
            // Summary Statistics
            out.println("\n" + "=".repeat(60))
            out.println("SUMMARY STATISTICS:")
            out.println("\n" + "=".repeat(60))
            if (results.isNotEmpty()) {
                val fitnessValues = results.map { it.fitness }
                val accuracyValues = results.map { it.accuracy }
                val featureReductions = results.filter { it.featuresSelected > 0 }.map { it.featureReduction }
                
                out.println("Fitness - Best: ${String.format("%.4f", fitnessValues.maxOrNull() ?: 0.0)}, " +
                           "Avg: ${String.format("%.4f", fitnessValues.average())}, " +
                           "Worst: ${String.format("%.4f", fitnessValues.minOrNull() ?: 0.0)}")
                
                out.println("Accuracy - Best: ${String.format("%.4f", accuracyValues.maxOrNull() ?: 0.0)}, " +
                           "Avg: ${String.format("%.4f", accuracyValues.average())}, " +
                           "Worst: ${String.format("%.4f", accuracyValues.minOrNull() ?: 0.0)}")
                
                if (featureReductions.isNotEmpty()) {
                    out.println("Feature Reduction - Best: ${String.format("%.1f", featureReductions.maxOrNull() ?: 0.0)}%, " +
                               "Avg: ${String.format("%.1f", featureReductions.average())}%, " +
                               "Worst: ${String.format("%.1f", featureReductions.minOrNull() ?: 0.0)}%")
                }
            }
        }
        
        println("Feature selection analysis report: ${reportFile.absolutePath}")
    }
    
    /**
     * Get experiment summary with feature selection metrics
     */
    fun getSummary(): String {
        val avgFeatures = if (results.any { it.featuresSelected > 0 }) {
            results.filter { it.featuresSelected > 0 }.map { it.featuresSelected }.average()
        } else 0.0
        
        val avgReduction = if (results.any { it.featuresSelected > 0 }) {
            results.filter { it.featuresSelected > 0 }.map { it.featureReduction }.average()
        } else 0.0
        
        return "L9 Taguchi Design: ${parameters.size} parameters, 9 configurations, ${results.size} total results" +
               if (avgFeatures > 0) " | Avg Features: ${String.format("%.1f", avgFeatures)}, Avg Reduction: ${String.format("%.1f", avgReduction)}%" else ""
    }
    
    /**
     * Get feature selection statistics
     */
    fun getFeatureStats(): Map<String, Double> {
        val validResults = results.filter { it.featuresSelected > 0 }
        return if (validResults.isNotEmpty()) {
            mapOf(
                "avgFeaturesSelected" to validResults.map { it.featuresSelected }.average(),
                "avgFeatureReduction" to validResults.map { it.featureReduction }.average(),
                "bestAccuracy" to (results.map { it.accuracy }.maxOrNull() ?: 0.0),
                "avgAccuracyImprovement" to results.filter { it.baselineAccuracy > 0 }.map { it.accuracyImprovement }.average(),
                "bestFeatureEfficiency" to (validResults.map { it.featureEfficiency }.maxOrNull() ?: 0.0)
            )
        } else {
            emptyMap()
        }
    }
    
    /**
     * Export detailed results to CSV
     */
    fun exportToCSV() {
        val timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss"))
        val csvFile = File("taguchi_feature_selection_results_$timestamp.csv")
        
        csvFile.printWriter().use { out ->
            out.println("experiment_id,population_size,max_iterations,mutation_rate,max_solutions," +
                       "fitness,accuracy,features_selected,total_features,feature_reduction_percent," +
                       "baseline_accuracy,accuracy_improvement,feature_efficiency,runtime_ms,selected_features")
            
            results.forEach { result ->
                val config = result.configuration
                val selectedFeaturesStr = result.selectedFeatureMask.joinToString(";")
                out.println("${config.experimentId}," +
                          "${config.parameters["populationSize"]}," +
                          "${config.parameters["maxIterations"]}," +
                          "${config.parameters["mutationRate"]}," +
                          "${config.parameters["maxSolutions"]}," +
                          "${result.fitness}," +
                          "${result.accuracy}," +
                          "${result.featuresSelected}," +
                          "${result.totalFeatures}," +
                          "${String.format("%.2f", result.featureReduction)}," +
                          "${result.baselineAccuracy}," +
                          "${String.format("%.4f", result.accuracyImprovement)}," +
                          "${String.format("%.6f", result.featureEfficiency)}," +
                          "${result.runtime}," +
                          "\"$selectedFeaturesStr\"")
            }
        }
        
        println("CSV exported: ${csvFile.absolutePath}")
    }
}
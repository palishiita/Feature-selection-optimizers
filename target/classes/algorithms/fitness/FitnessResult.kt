package com.technosudo.algorithms.fitness

import com.technosudo.evaluation.EvaluationMetrics

data class FitnessResult(
    val fitness: Double,
    val metrics: EvaluationMetrics
)

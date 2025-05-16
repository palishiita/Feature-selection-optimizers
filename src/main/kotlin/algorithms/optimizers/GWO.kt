package com.technosudo.algorithms.optimizers

import com.technosudo.algorithms.fitness.FitnessFunction
import org.jetbrains.kotlinx.dataframe.DataFrame

class GWO(
    override val populationSize: Int = 30,
    override val maxIterations: Int = 100,
//    private val transferFunction: TransferFunction = SigmoidTransferFunction()
) : Optimizer {

    override val name: String  = "Binary Grey Wolf Optimizer"

    override fun optimize(
        dataset: DataFrame<*>,
        fitnessFunction: FitnessFunction
    ): DataFrame<*> {
        TODO("Logic to be implemented")
    }
}
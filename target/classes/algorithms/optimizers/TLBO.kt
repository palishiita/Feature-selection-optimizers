package algorithms.binary;

import com.technosudo.algorithms.optimizers.Optimizer
import com.technosudo.algorithms.fitness.FitnessFunction
import org.jetbrains.kotlinx.dataframe.DataFrame

class TLBO(
    override val populationSize: Int = 30,
    override val maxIterations: Int = 100,
) : Optimizer {

    override val name: String  = "Binary Grey Wolf Optimizer"

    override fun optimize(
        dataset: DataFrame<*>,
        fitnessFunction: FitnessFunction
    ): DataFrame<*> {
        TODO("Logic to be implemented")
    }
}

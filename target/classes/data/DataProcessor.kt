package com.technosudo.data

import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.maxBy
import org.jetbrains.kotlinx.dataframe.api.minBy
import org.jetbrains.kotlinx.dataframe.api.update
import org.jetbrains.kotlinx.dataframe.name
import org.jetbrains.kotlinx.dataframe.type
import kotlin.reflect.typeOf

object DataProcessor {

    fun DataFrame<*>.minMaxNormalize(): DataFrame<*> {
        for (column in this.columns()) {
            if (column.type == typeOf<Number>())
                this.update(column.name) { valueAny ->
                    val value = (valueAny as Number).toDouble()
                    val max = (column.maxBy(skipNaN = true) { (it as Number).toDouble() } as Number).toDouble()
                    val min = (column.minBy(skipNaN = true) { (it as Number).toDouble() } as Number).toDouble()
                    (value - min) / (max - min)
            }
        }
        return this
    }
}
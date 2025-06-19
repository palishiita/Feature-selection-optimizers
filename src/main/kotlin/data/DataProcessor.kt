package com.technosudo.data

import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.convert
import org.jetbrains.kotlinx.dataframe.api.rename
import org.jetbrains.kotlinx.dataframe.api.update

object DataProcessor {

    fun DataFrame<*>.minMaxNormalize(): DataFrame<*> {
        var normalized = this

        for (column in this.columns()) {
            if (column.values().all { it is Number }) {
                val colName = column.name()

                // First convert the column to Double type
                normalized = normalized.convert(colName).to<Double>()

                val numericValues = normalized[colName].values().filterIsInstance<Double>()
                val max = numericValues.maxOrNull() ?: continue
                val min = numericValues.minOrNull() ?: continue

                normalized = normalized.update(colName) { value ->
                    val num = (value as Double)
                    if (max != min) (num - min) / (max - min) else 0.0
                }
            }
        }
        return normalized
    }

    fun DataFrame<*>.nameColumns(): DataFrame<*> {
        this.columnNames()
            .mapIndexed { index, name ->
                this.rename(name).to(index) }
        return this
    }
}
package com.technosudo.data

import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.convert
import org.jetbrains.kotlinx.dataframe.api.count
import org.jetbrains.kotlinx.dataframe.api.dropNulls
import org.jetbrains.kotlinx.dataframe.api.remove
import org.jetbrains.kotlinx.dataframe.api.rename
import org.jetbrains.kotlinx.dataframe.api.sumOf
import org.jetbrains.kotlinx.dataframe.api.update
import org.jetbrains.kotlinx.dataframe.size

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

    fun DataFrame<*>.castToDouble(): DataFrame<*> =
        this.convert(*columnNames().toTypedArray()).to<Double?>()



    fun DataFrame<*>.nameColumns(): DataFrame<*> {
        this.columnNames()
            .mapIndexed { index, name ->
                this.rename(name).to(index) }
        return this
    }

    fun DataFrame<*>.removeColForNullShare(threshold: Double): DataFrame<*> {
        val columnsToRemove = this.columns()
            .filter { col ->
                val nullCount = col.count { it == null }
                threshold < nullCount / col.size}
            .map { it.name() }

        println("\nColumns to remove (nulls > ${threshold * 100}%): $columnsToRemove")
        return this.remove(*columnsToRemove.toTypedArray())
    }

    fun DataFrame<*>.removeColIfNullPresent(): DataFrame<*> {
        val columnsToRemove = this.columns()
            .filter { col -> col.hasNulls() }
            .map { it.name() }

        println("\nColumns to remove : $columnsToRemove")
        return this.remove(*columnsToRemove.toTypedArray())
    }
}
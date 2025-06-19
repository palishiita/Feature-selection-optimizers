package com.technosudo.misc

import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.columns.ColumnGroup
import org.jetbrains.kotlinx.dataframe.columns.FrameColumn
import org.jetbrains.kotlinx.dataframe.columns.ValueColumn
import smile.data.Tuple
import smile.data.type.DataType
import smile.data.type.StructField
import smile.data.type.StructType
import smile.data.vector.DoubleVector
import smile.data.vector.ValueVector
import kotlin.Double.Companion.NaN

object Mapper {

    fun DataFrame<*>.toSmileDF(): smile.data.DataFrame {
        val smileColumnsData = mutableListOf<List<Double>>()
        val smileColumnNames = mutableListOf<String>()
        val smileSchemaFields = mutableListOf<StructField>()

        this.columns().forEach { col ->
            val columnName = col.name()
            val columnType = col.type()

            when (col) {
                is ValueColumn<*> -> {
                    if (columnType.classifier != Double::class)
                        throw Exception("Not Double type")

                    val doubleValues = col.toList().map { it as? Double ?: NaN }
                    smileColumnsData.add(doubleValues)
                    smileColumnNames.add(columnName)
                    smileSchemaFields.add(StructField(columnName, DataType.of(Double::class.java)))
                }
                else -> println("Warning: Complex column type ('$columnName': $columnType) cannot be directly converted to Smile DataFrame. Skipping or converting as ObjectVector.")
            }
        }

        val smileSchema = StructType(smileSchemaFields)

        val numRows = this.rowsCount()
        val smileTuples = mutableListOf<Tuple>()

        for (i in 0 until numRows) {
            val rowData = smileColumnNames.mapIndexed { colIndex, _ ->
                smileColumnsData[colIndex][i]
            }.toTypedArray()
            smileTuples.add(Tuple.of(rowData))
        }

        return smile.data.DataFrame.of(smileSchema, smileTuples)
    }
}
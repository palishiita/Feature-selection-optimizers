package com.technosudo.data

import com.technosudo.data.DataProcessor.castToDouble
import com.technosudo.data.DataProcessor.minMaxNormalize
import com.technosudo.data.DataProcessor.nameColumns
import com.technosudo.data.DataProcessor.removeColForNullShare
import com.technosudo.data.DataProcessor.removeColIfNullPresent
import org.jetbrains.kotlinx.dataframe.DataColumn
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.*
import org.jetbrains.kotlinx.dataframe.io.read

interface DataLoader : Iterable<Pair<DataFrame<*>, DataColumn<*>>> {
    companion object {
        fun bcw(): DataLoader = BCW()
        fun arrhythmia(): DataLoader = Arrhythmia()
        fun semiConductor(): DataLoader = SemiConductor()
    }
}

private class BCW : DataLoader {
    override fun iterator(): Iterator<Pair<DataFrame<*>, DataColumn<*>>> {
        return object : Iterator<Pair<DataFrame<*>, DataColumn<*>>> {
            var current: String? = "src/main/kotlin/data/datasets/breast-cancer-wisconsin/wdbc.data"

            override fun hasNext(): Boolean = current != null
            override fun next(): Pair<DataFrame<*>, DataColumn<*>> = current?.let {
                current = null
                val all = DataFrame.read(it).dropNulls()

                val target = all.getColumn(1).map { type ->
                    when (type.toString().trim()) {
                        "B" -> 0.0
                        "M" -> 1.0
                        else -> {
                            println("Unknown label value: '${type}'")
                            0.0
                        }
                    }
                }



                val train = all.select { cols().drop(2) }.minMaxNormalize()
                Pair(train, target)
            } ?: throw NoSuchElementException()
        }
    }
}

private class Arrhythmia : DataLoader {
    override fun iterator(): Iterator<Pair<DataFrame<*>, DataColumn<*>>> {
        return object : Iterator<Pair<DataFrame<*>, DataColumn<*>>> {
            var current: String? = "src/main/kotlin/data/datasets/arrhythmia/arrhythmia.data"

            override fun hasNext(): Boolean = current != null
            override fun next(): Pair<DataFrame<*>, DataColumn<*>> = current?.let {
                current = null
                val df = DataFrame.read(it).dropNulls()
                val lastColName = df.columns().last().name()
                val features = df.select { cols().dropLast(1) }.minMaxNormalize()
                val labels = df[lastColName].convertTo<Double>()
                Pair(features, labels)
            } ?: throw NoSuchElementException()
        }
    }
}

private class SemiConductor : DataLoader {
    override fun iterator(): Iterator<Pair<DataFrame<*>, DataColumn<*>>> {
        return object : Iterator<Pair<DataFrame<*>, DataColumn<*>>> {
            var current: String? = "src/main/kotlin/data/datasets/semi-conductor/data.csv"

            override fun hasNext(): Boolean = current != null
            override fun next(): Pair<DataFrame<*>, DataColumn<*>> = current?.let {
                current = null
                val df = DataFrame
                    .read(it)
                    .remove{ col(0) }
                    .castToDouble()
                    .fillNulls { colsOf<Double>() }
                    .with { 0.0 }
                val features = df
                    .select { cols().dropLast(1) }
                    .minMaxNormalize()
                    .nameColumns()
                val labels = df["Pass/Fail"].convertTo<Double>()
                Pair(features, labels)
            } ?: throw NoSuchElementException()
        }
    }
}

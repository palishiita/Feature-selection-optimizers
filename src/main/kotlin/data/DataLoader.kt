package com.technosudo.data

import com.technosudo.data.DataProcessor.minMaxNormalize
import org.jetbrains.kotlinx.dataframe.DataColumn
import org.jetbrains.kotlinx.dataframe.DataFrame
import org.jetbrains.kotlinx.dataframe.api.convertTo
import org.jetbrains.kotlinx.dataframe.api.dropNulls
import org.jetbrains.kotlinx.dataframe.api.getColumn
import org.jetbrains.kotlinx.dataframe.api.getColumns
import org.jetbrains.kotlinx.dataframe.api.map
import org.jetbrains.kotlinx.dataframe.api.maxBy
import org.jetbrains.kotlinx.dataframe.api.select
import org.jetbrains.kotlinx.dataframe.io.read

interface DataLoader: Iterable<Pair<DataFrame<*>, DataColumn<*>>> {
    companion object {
        fun bcw(): DataLoader = BCW()
//        fun arrhythmia(): DataLoader = Arrhythmia()
//        fun aml(): DataLoader = BCW()
    }
}

private class BCW(): DataLoader {
    override fun iterator(): Iterator<Pair<DataFrame<*>, DataColumn<*>>> {
        return object : Iterator<Pair<DataFrame<*>, DataColumn<*>>> {
            var current: String? = "datasets/breast-cancer-wisconsin/wdbc.data"

            override fun hasNext(): Boolean = current != null
            override fun next(): Pair<DataFrame<*>, DataColumn<*>> = current?.let {
                current = null

                val all = DataFrame.read(path = it).dropNulls()
                val target = all.getColumn(1).map { type -> when(type) {
                    "B" -> 0.0
                    "M" -> 1.0
                    else -> 0.0
                } }
                val train = all.select { cols().drop(2) }.minMaxNormalize()
                Pair(train, target)
            } ?: throw NoSuchElementException()
        }
    }
}
//private class Arrhythmia(): DataLoader {
//    override fun iterator(): Iterator<DataFrame<*>> {
//        return object : Iterator<DataFrame<*>> {
//            var current: String? = "datasets/arrhythmia/arrhythmia.data"
//
//            override fun hasNext(): Boolean = current != null
//            override fun next(): DataFrame<*> = current?.let {
//                current = null
//                DataFrame.read(path = it)
//            } ?: throw NoSuchElementException()
//        }
//    }
//}
//private class AML(): DataLoader {
//    override fun load(): Collection<DataFrame<*>> = TODO()
//}
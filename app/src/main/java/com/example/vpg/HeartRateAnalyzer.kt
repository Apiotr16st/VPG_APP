package com.example.vpg

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

data class ChannelState(
    val timestamps: MutableList<Long> = mutableListOf(),
    val samples: MutableList<Double> = mutableListOf(),
    val rollingBpmBuffer: MutableList<Double> = mutableListOf(),
    var currentBpm: Double? = null,
    var rollingBpmAverage: Double? = null,
    var lastBatchFps: Double? = null,
    var processedBatchSamples: Int = 0,
)

class HeartRateAnalyzer {

    val bufferSize = 50
    private val minBpm = 55.0
    private val maxBpm = 200.0
    private val rollingBufferSize = 10
    private val filterPadLength = 21
    private val butterworthB = doubleArrayOf(
        0.01036812956893816,
        0.0,
        -0.03110438870681449,
        0.0,
        0.03110438870681449,
        0.0,
        -0.01036812956893816,
    )
    private val butterworthA = doubleArrayOf(
        1.0,
        -4.65859829168493,
        9.326603288552544,
        -10.29243302656037,
        6.6129908351859426,
        -2.3462622197511407,
        0.35918862147425845,
    )

    val channelG = ChannelState()

    fun processSamples(timestamp: Long, yVal: Float, gVal: Float) {
        updateChannel(channelG, timestamp, gVal.toDouble())
    }

    private fun updateChannel(state: ChannelState, timestamp: Long, value: Double) {
        state.timestamps.add(timestamp)
        state.samples.add(value)

        val batchEnd = state.processedBatchSamples + bufferSize
        if (state.samples.size >= batchEnd) {
            val batchSamples = state.samples.subList(state.processedBatchSamples, batchEnd).toList()
            val batchTimestamps = state.timestamps.subList(state.processedBatchSamples, batchEnd).toList()
            state.processedBatchSamples = batchEnd
            estimateBpmByAutocorrelation(state, batchSamples, batchTimestamps)
        }
    }

    private fun estimateBpmByAutocorrelation(
        state: ChannelState,
        samples: List<Double>,
        timestamps: List<Long>,
    ) {
        val timeSpanSeconds = (timestamps.last() - timestamps.first()) / 1000.0
        if (timeSpanSeconds <= 0.0) return

        val actualFps = (samples.size - 1) / timeSpanSeconds
        state.lastBatchFps = actualFps
        val filtered = preprocessForHeartRate(samples, actualFps) ?: return
        val autocorr = autocorrelate(filtered)
        val maxAbs = autocorr.maxOfOrNull { abs(it) } ?: return
        if (maxAbs <= 1e-9) return

        for (i in autocorr.indices) {
            autocorr[i] /= maxAbs
        }

        val minLag = max(1, floor(actualFps / (maxBpm / 60.0)).toInt())
        val maxLag = min(autocorr.lastIndex, ceil(actualFps / (minBpm / 60.0)).toInt())
        if (maxLag <= minLag) return

        val bestLag = findBestLag(autocorr, minLag, maxLag, actualFps)
        val calculatedBpm = (actualFps / bestLag) * 60.0
        if (calculatedBpm !in minBpm..maxBpm) return

        state.currentBpm = calculatedBpm
        state.rollingBpmBuffer.add(calculatedBpm)
        if (state.rollingBpmBuffer.size > rollingBufferSize) {
            state.rollingBpmBuffer.removeAt(0)
        }
        state.rollingBpmAverage = if (state.rollingBpmBuffer.size == rollingBufferSize) {
            state.rollingBpmBuffer.average()
        } else {
            null
        }
    }

    private fun preprocessForHeartRate(samples: List<Double>, actualFps: Double): DoubleArray? {
        val nyquist = actualFps / 2.0
        val lowCutHz = minBpm / 60.0
        val highCutHz = min(maxBpm / 60.0, nyquist * 0.95)
        if (lowCutHz <= 0.0 || lowCutHz >= nyquist || highCutHz <= lowCutHz) return null

        val detrended = detrendLinear(samples)
        return applyScipyButterworthForwardBackward(detrended)
    }

    private fun detrendLinear(samples: List<Double>): DoubleArray {
        val n = samples.size
        val xMean = (n - 1) / 2.0
        val yMean = samples.average()

        var numerator = 0.0
        var denominator = 0.0
        for (i in 0 until n) {
            val x = i - xMean
            numerator += x * (samples[i] - yMean)
            denominator += x * x
        }

        val slope = if (denominator > 0.0) numerator / denominator else 0.0
        val intercept = yMean - slope * xMean
        return DoubleArray(n) { i -> samples[i] - (slope * i + intercept) }
    }

    private fun applyScipyButterworthForwardBackward(values: DoubleArray): DoubleArray {
        if (values.isEmpty()) return values

        val padLength = min(filterPadLength, values.size - 1)
        val padded = oddPad(values, padLength)
        val forward = applyIirFilter(padded, butterworthB, butterworthA)
        forward.reverse()
        val backward = applyIirFilter(forward, butterworthB, butterworthA)
        backward.reverse()

        return backward.copyOfRange(padLength, padLength + values.size)
    }

    private fun oddPad(values: DoubleArray, padLength: Int): DoubleArray {
        if (padLength <= 0) return values.copyOf()

        val padded = DoubleArray(values.size + 2 * padLength)
        val first = values.first()
        val last = values.last()

        for (i in 0 until padLength) {
            padded[i] = 2.0 * first - values[padLength - i]
        }
        values.copyInto(padded, destinationOffset = padLength)
        for (i in 0 until padLength) {
            padded[padLength + values.size + i] = 2.0 * last - values[values.lastIndex - 1 - i]
        }

        return padded
    }

    private fun applyIirFilter(input: DoubleArray, b: DoubleArray, a: DoubleArray): DoubleArray {
        val output = DoubleArray(input.size)

        for (i in input.indices) {
            var value = 0.0
            for (j in b.indices) {
                val inputIndex = i - j
                if (inputIndex >= 0) {
                    value += b[j] * input[inputIndex]
                }
            }
            for (j in 1 until a.size) {
                val outputIndex = i - j
                if (outputIndex >= 0) {
                    value -= a[j] * output[outputIndex]
                }
            }
            output[i] = value / a[0]
        }

        return output
    }

    private fun autocorrelate(values: DoubleArray): DoubleArray {
        val result = DoubleArray(values.size)
        for (lag in values.indices) {
            var sum = 0.0
            for (i in 0 until values.size - lag) {
                sum += values[i] * values[i + lag]
            }
            result[lag] = sum
        }
        return result
    }

    private fun findBestLag(autocorr: DoubleArray, minLag: Int, maxLag: Int, actualFps: Double): Int {
        val minPeakDistance = max(1, (actualFps * 0.25).toInt())
        val peaks = mutableListOf<Int>()

        var lastPeak = -minPeakDistance
        for (lag in minLag + 1 until maxLag) {
            if (lag - lastPeak < minPeakDistance) continue
            if (autocorr[lag] > autocorr[lag - 1] && autocorr[lag] >= autocorr[lag + 1]) {
                peaks.add(lag)
                lastPeak = lag
            }
        }

        if (peaks.isEmpty()) {
            return (minLag..maxLag).maxBy { autocorr[it] }
        }

        val maxPeakValue = peaks.maxOf { autocorr[it] }
        val strongPeakThreshold = maxPeakValue * 0.75
        return peaks.firstOrNull { autocorr[it] >= strongPeakThreshold }
            ?: peaks.maxBy { autocorr[it] }
    }

}

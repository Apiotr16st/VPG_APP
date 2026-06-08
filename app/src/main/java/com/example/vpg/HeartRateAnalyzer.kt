package com.example.vpg

import org.apache.commons.math3.transform.DftNormalization
import org.apache.commons.math3.transform.FastFourierTransformer
import org.apache.commons.math3.transform.TransformType
import kotlin.math.pow
import kotlin.math.sqrt

data class ChannelState(
    val timestamps: MutableList<Long> = mutableListOf(),
    val samples: MutableList<Double> = mutableListOf(),
    val rollingBpmBuffer: MutableList<Double> = mutableListOf(),
    var currentBpm: Double? = null,
    var rollingBpmAverage: Double? = null
)

class HeartRateAnalyzer {

    private val bufferSize = 128
    private val minBpm = 48.0
    private val maxBpm = 150.0
    private val rollingBufferSize = 5

    val channelG = ChannelState()

    fun processSamples(timestamp: Long, yVal: Float, gVal: Float) {
        updateChannel(channelG, timestamp, gVal.toDouble())
    }

    private fun updateChannel(state: ChannelState, timestamp: Long, value: Double) {
        state.timestamps.add(timestamp)
        state.samples.add(value)

        if (state.samples.size > bufferSize) {
            state.samples.removeAt(0)
            state.timestamps.removeAt(0)
        }

        if (state.samples.size == bufferSize) {
            calculateBpmFFT(state)
        }
    }

    private fun calculateBpmFFT(state: ChannelState) {
        val timeSpanSeconds = (state.timestamps.last() - state.timestamps.first()) / 1000.0
        if (timeSpanSeconds <= 0.0) return
        val currentFps = (bufferSize - 1) / timeSpanSeconds

        // 1. Detrending (usunięcie stałej składowej)
        val mean = state.samples.average()
        val detrended = DoubleArray(bufferSize) { i -> state.samples[i] - mean }

        // 2. Okno Hamminga (wygładza końce sygnału)
        val windowed = DoubleArray(bufferSize)
        for (i in 0 until bufferSize) {
            val hammingWindow = 0.54 - 0.46 * Math.cos((2.0 * Math.PI * i) / (bufferSize - 1))
            windowed[i] = detrended[i] * hammingWindow
        }

        // 3. Szybka Transformata Fouriera (FFT)
        val transformer = FastFourierTransformer(DftNormalization.STANDARD)
        val complexData = transformer.transform(windowed, TransformType.FORWARD)

        // 4. Szukanie najwyższego szczytu tylko w paśmie 48-150 BPM
        var maxMagnitude = -1.0
        var bestFreqIndex = -1

        for (i in 1 until bufferSize / 2) {
            val frequency = (i * currentFps) / bufferSize
            val bpmAtFreq = frequency * 60.0

            if (bpmAtFreq in minBpm..maxBpm) {
                val real = complexData[i].real
                val imaginary = complexData[i].imaginary
                val magnitude = sqrt(real.pow(2.0) + imaginary.pow(2.0))

                if (magnitude > maxMagnitude) {
                    maxMagnitude = magnitude
                    bestFreqIndex = i
                }
            }
        }

        // 5. Zapis wyniku
        if (bestFreqIndex != -1) {
            val bestFrequency = (bestFreqIndex * currentFps) / bufferSize
            val calculatedBpm = bestFrequency * 60.0

            state.currentBpm = calculatedBpm
            state.rollingBpmBuffer.add(calculatedBpm)

            if (state.rollingBpmBuffer.size > rollingBufferSize) {
                state.rollingBpmBuffer.removeAt(0)
            }
            state.rollingBpmAverage = state.rollingBpmBuffer.average()
        }
    }
}
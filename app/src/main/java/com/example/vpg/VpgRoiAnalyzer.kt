package com.example.vpg

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Rect
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import kotlin.math.max
import kotlin.math.min

data class VpgResult(val meanY: Float, val meanG: Float, val roiRect: Rect)

class VpgRoiAnalyzer {

    private val roiSmoothingAlpha = 0.35f
    private var smoothedRoi: FloatArray? = null

    fun processFrame(result: FaceLandmarkerResult, bitmap: Bitmap): VpgResult? {
        if (result.faceLandmarks().isEmpty()) return null

        val landmarks = result.faceLandmarks()[0]
        val frameW = bitmap.width
        val frameH = bitmap.height
        val rawRoi = getForeheadRoiCoordinates(landmarks, frameW, frameH)

        val previousRoi = smoothedRoi
        val currentRoi = if (previousRoi == null) {
            rawRoi
        } else {
            floatArrayOf(
                roiSmoothingAlpha * rawRoi[0] + (1.0f - roiSmoothingAlpha) * previousRoi[0],
                roiSmoothingAlpha * rawRoi[1] + (1.0f - roiSmoothingAlpha) * previousRoi[1],
                roiSmoothingAlpha * rawRoi[2] + (1.0f - roiSmoothingAlpha) * previousRoi[2],
                roiSmoothingAlpha * rawRoi[3] + (1.0f - roiSmoothingAlpha) * previousRoi[3],
            )
        }
        smoothedRoi = currentRoi

        val rect = clampRoi(currentRoi[0], currentRoi[1], currentRoi[2], currentRoi[3], frameW, frameH)
        if (rect.width() <= 0 || rect.height() <= 0) return null

        val means = calculateRoiMeans(bitmap, rect)
        return VpgResult(means.first, means.second, rect)
    }

    private fun getForeheadRoiCoordinates(
        landmarks: List<NormalizedLandmark>,
        frameW: Int,
        frameH: Int,
    ): FloatArray {
        var faceXMin = Float.POSITIVE_INFINITY
        var faceYMin = Float.POSITIVE_INFINITY
        var faceXMax = Float.NEGATIVE_INFINITY
        var faceYMax = Float.NEGATIVE_INFINITY

        for (landmark in landmarks) {
            val x = landmark.x() * frameW
            val y = landmark.y() * frameH
            faceXMin = min(faceXMin, x)
            faceYMin = min(faceYMin, y)
            faceXMax = max(faceXMax, x)
            faceYMax = max(faceYMax, y)
        }

        val faceW = max(1.0f, faceXMax - faceXMin)
        val faceH = max(1.0f, faceYMax - faceYMin)
        val browIndexes = intArrayOf(70, 63, 105, 66, 107, 336, 296, 334, 293, 300)

        var browCenterX = 0.0f
        var browY = 0.0f
        for (index in browIndexes) {
            browCenterX += landmarks[index].x() * frameW
            browY += landmarks[index].y() * frameH
        }
        browCenterX /= browIndexes.size
        browY /= browIndexes.size

        val foreheadTopY = landmarks[10].y() * frameH
        val foreheadSpan = max(8.0f, browY - foreheadTopY)

        val roiW = 0.36f * faceW
        var roiH = min(0.36f * faceH, foreheadSpan)
        val roiX = browCenterX - roiW / 2.0f + 0.03f * faceW
        val roiY = foreheadTopY + 0.12f * foreheadSpan

        val browMargin = 0.22f * foreheadSpan
        val maxRoiBottom = browY - browMargin
        if (roiY + roiH > maxRoiBottom) {
            roiH = max(6.0f, maxRoiBottom - roiY)
        }

        return floatArrayOf(roiX, roiY, roiW, roiH)
    }

    private fun clampRoi(x: Float, y: Float, w: Float, h: Float, frameW: Int, frameH: Int): Rect {
        val x1 = max(0, min(frameW - 1, Math.round(x)))
        val y1 = max(0, min(frameH - 1, Math.round(y)))
        val x2 = max(0, min(frameW, Math.round(x + max(0.0f, w))))
        val y2 = max(0, min(frameH, Math.round(y + max(0.0f, h))))
        return Rect(x1, y1, x2, y2)
    }

    private fun calculateRoiMeans(bitmap: Bitmap, rect: Rect): Pair<Float, Float> {
        var sumY = 0.0
        var sumG = 0.0
        val pixelCount = rect.width() * rect.height()
        val pixels = IntArray(pixelCount)

        bitmap.getPixels(pixels, 0, rect.width(), rect.left, rect.top, rect.width(), rect.height())

        for (pixel in pixels) {
            val r = Color.red(pixel)
            val g = Color.green(pixel)
            val b = Color.blue(pixel)
            val y = 0.299f * r + 0.587f * g + 0.114f * b

            sumY += y
            sumG += g
        }

        return Pair((sumY / pixelCount).toFloat(), (sumG / pixelCount).toFloat())
    }
}

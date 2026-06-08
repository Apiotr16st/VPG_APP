package com.example.vpg

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Rect
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import kotlin.math.max
import kotlin.math.min


data class VpgResult(val meanY: Float, val meanG: Float, val roiRect: Rect)
class VpgRoiAnalyzer {

    private val roiSmoothingAlpha = 0.15f
    private var smoothedRoi: FloatArray? = null

    fun processFrame(result: FaceLandmarkerResult, bitmap: Bitmap): VpgResult? {
        if (result.faceLandmarks().isEmpty()) return null

        val landmarks = result.faceLandmarks()[0]
        val frameW = bitmap.width
        val frameH = bitmap.height

        val rawRoi = getForeheadRoiCoordinates(landmarks, frameW, frameH)

        val currentRoi = if (smoothedRoi == null) {
            rawRoi
        } else {
            floatArrayOf(
                roiSmoothingAlpha * rawRoi[0] + (1 - roiSmoothingAlpha) * smoothedRoi!![0],
                roiSmoothingAlpha * rawRoi[1] + (1 - roiSmoothingAlpha) * smoothedRoi!![1],
                roiSmoothingAlpha * rawRoi[2] + (1 - roiSmoothingAlpha) * smoothedRoi!![2],
                roiSmoothingAlpha * rawRoi[3] + (1 - roiSmoothingAlpha) * smoothedRoi!![3]
            )
        }
        smoothedRoi = currentRoi

        val rect = clampRoi(currentRoi[0], currentRoi[1], currentRoi[2], currentRoi[3], frameW, frameH)

        if (rect.width() > 0 && rect.height() > 0) {
            val means = calculateRoiMeans(bitmap, rect)
            // ZWRACAMY TERAZ 3 RZECZY: Jasność, Zieleń i dokładny Prostokąt
            return VpgResult(means.first, means.second, rect)
        }
        return null
    }

    private fun getForeheadRoiCoordinates(landmarks: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>, frameW: Int, frameH: Int): FloatArray {
        // Landmark 10 to góra czoła (linia włosów), 151 to środek czoła, 9 to między brwiami
        val foreheadTop = landmarks[10]
        val foreheadMid = landmarks[151]
        val betweenBrows = landmarks[9]

        // Szerokość i wysokość twarzy na podstawie punktów charakterystycznych
        // 234 i 454 to skrajne punkty kości policzkowych/uszu
        val faceLeft = landmarks[234].x() * frameW
        val faceRight = landmarks[454].x() * frameW
        val faceTop = landmarks[10].y() * frameH
        val faceBottom = landmarks[152].y() * frameH // 152 to podbródek

        val faceW = Math.abs(faceRight - faceLeft)
        val faceH = Math.abs(faceBottom - faceTop)

        // Wyliczamy wysokość czoła (od linii włosów do brwi)
        val browY = betweenBrows.y() * frameH
        val topY = foreheadTop.y() * frameH
        val foreheadSpan = Math.max(20.0f, browY - topY)

        // Definiujemy ROI na środku czoła
        val roiW = 0.50f * faceW  // Szerokość: 50% szerokości twarzy
        val roiH = 0.40f * foreheadSpan // Wysokość: 40% wysokości czoła

        // Środek czoła (X) i nieco powyżej brwi (Y)
        val roiX = (foreheadMid.x() * frameW) - (roiW / 2f)
        val roiY = (foreheadMid.y() * frameH) - (roiH / 2f)

        return floatArrayOf(roiX, roiY, roiW, roiH)
    }

    private fun clampRoi(x: Float, y: Float, w: Float, h: Float, frameW: Int, frameH: Int): Rect {
        val x1 = max(0, min(frameW - 1, Math.round(x)))
        val y1 = max(0, min(frameH - 1, Math.round(y)))
        val x2 = max(0, min(frameW, Math.round(x + w)))
        val y2 = max(0, min(frameH, Math.round(y + h)))
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
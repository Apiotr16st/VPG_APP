package com.example.vpg.ui.camera

import android.graphics.Rect
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceLandmark

internal fun Face.toOverlayData(imageWidth: Int, imageHeight: Int): FaceOverlayData {
    val faceRect = boundingBox.toRectF()
    val leftEye = getLandmark(FaceLandmark.LEFT_EYE)?.position
    val rightEye = getLandmark(FaceLandmark.RIGHT_EYE)?.position

    val foreheadRect = if (leftEye != null && rightEye != null) {
        val eyeMidX = (leftEye.x + rightEye.x) / 2f
        val eyeMidY = (leftEye.y + rightEye.y) / 2f
        val interEyeDistance =
            kotlin.math.abs(rightEye.x - leftEye.x).coerceAtLeast(faceRect.width() * 0.18f)
        val foreheadWidth = (interEyeDistance * 1.05f)
            .coerceIn(faceRect.width() * 0.18f, faceRect.width() * 0.30f)
        val foreheadHeight = (faceRect.height() * 0.11f)
            .coerceIn(faceRect.height() * 0.09f, faceRect.height() * 0.14f)
        val foreheadCenterY = (eyeMidY - faceRect.height() * 0.20f)
            .coerceIn(
                faceRect.top + faceRect.height() * 0.12f,
                faceRect.top + faceRect.height() * 0.26f
            )

        RectFCompat(
            left = eyeMidX - foreheadWidth / 2f,
            top = foreheadCenterY - foreheadHeight / 2f,
            right = eyeMidX + foreheadWidth / 2f,
            bottom = foreheadCenterY + foreheadHeight / 2f
        ).clampedInside(
            faceRect.inset(
                horizontalFraction = 0.08f,
                topFraction = 0.06f,
                bottomFraction = 0.52f
            )
        )
    } else {
        val foreheadWidth = faceRect.width() * 0.22f
        val foreheadHeight = faceRect.height() * 0.11f
        val foreheadCenterX = faceRect.centerX()
        val foreheadCenterY = faceRect.top + faceRect.height() * 0.19f

        RectFCompat(
            left = foreheadCenterX - foreheadWidth / 2f,
            top = foreheadCenterY - foreheadHeight / 2f,
            right = foreheadCenterX + foreheadWidth / 2f,
            bottom = foreheadCenterY + foreheadHeight / 2f
        ).clampedInside(
            faceRect.inset(
                horizontalFraction = 0.10f,
                topFraction = 0.08f,
                bottomFraction = 0.54f
            )
        )
    }

    return FaceOverlayData(
        boundingBox = faceRect,
        foreheadRect = foreheadRect,
        imageWidth = imageWidth.toFloat(),
        imageHeight = imageHeight.toFloat()
    ).stabilized()
}

internal fun mapRectToPreview(
    rect: RectFCompat,
    imageWidth: Float,
    imageHeight: Float,
    previewWidth: Float,
    previewHeight: Float,
    mirrorHorizontally: Boolean
): RectFCompat {
    val scale = maxOf(previewWidth / imageWidth, previewHeight / imageHeight)
    val scaledWidth = imageWidth * scale
    val scaledHeight = imageHeight * scale
    val dx = (previewWidth - scaledWidth) / 2f
    val dy = (previewHeight - scaledHeight) / 2f

    val scaledLeft = rect.left * scale + dx
    val scaledTop = rect.top * scale + dy
    val scaledRight = rect.right * scale + dx
    val scaledBottom = rect.bottom * scale + dy

    return if (mirrorHorizontally) {
        RectFCompat(
            left = previewWidth - scaledRight,
            top = scaledTop,
            right = previewWidth - scaledLeft,
            bottom = scaledBottom
        )
    } else {
        RectFCompat(
            left = scaledLeft,
            top = scaledTop,
            right = scaledRight,
            bottom = scaledBottom
        )
    }
}

private fun Rect.toRectF(): RectFCompat {
    return RectFCompat(
        left = left.toFloat(),
        top = top.toFloat(),
        right = right.toFloat(),
        bottom = bottom.toFloat()
    )
}

internal data class FaceOverlayData(
    val boundingBox: RectFCompat,
    val foreheadRect: RectFCompat,
    val imageWidth: Float,
    val imageHeight: Float
) {
    fun stabilized(alpha: Float = 0.72f): FaceOverlayData {
        val previous = previousByImageSize[imageWidth to imageHeight]
        val smoothed = if (previous == null) {
            this
        } else {
            copy(
                boundingBox = previous.boundingBox.lerpTo(boundingBox, alpha),
                foreheadRect = previous.foreheadRect.lerpTo(foreheadRect, alpha)
            )
        }
        previousByImageSize[imageWidth to imageHeight] = smoothed
        return smoothed
    }

    companion object {
        private val previousByImageSize = mutableMapOf<Pair<Float, Float>, FaceOverlayData>()

        fun clearHistory() {
            previousByImageSize.clear()
        }
    }
}

internal data class RectFCompat(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
) {
    fun width(): Float = right - left
    fun height(): Float = bottom - top
    fun centerX(): Float = (left + right) / 2f

    fun lerpTo(target: RectFCompat, alpha: Float): RectFCompat {
        return RectFCompat(
            left = left + (target.left - left) * alpha,
            top = top + (target.top - top) * alpha,
            right = right + (target.right - right) * alpha,
            bottom = bottom + (target.bottom - bottom) * alpha
        )
    }

    fun clampedInside(container: RectFCompat): RectFCompat {
        val rectWidth = width().coerceAtMost(container.width())
        val rectHeight = height().coerceAtMost(container.height())
        val clampedLeft = left.coerceIn(container.left, container.right - rectWidth)
        val clampedTop = top.coerceIn(container.top, container.bottom - rectHeight)
        return RectFCompat(
            left = clampedLeft,
            top = clampedTop,
            right = clampedLeft + rectWidth,
            bottom = clampedTop + rectHeight
        )
    }

    fun inset(
        horizontalFraction: Float = 0f,
        topFraction: Float = 0f,
        bottomFraction: Float = 0f
    ): RectFCompat {
        val insetX = width() * horizontalFraction
        return RectFCompat(
            left = left + insetX,
            top = top + height() * topFraction,
            right = right - insetX,
            bottom = bottom - height() * bottomFraction
        )
    }
}

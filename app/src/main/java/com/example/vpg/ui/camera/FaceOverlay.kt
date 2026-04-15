package com.example.vpg.ui.camera

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp

@Composable
internal fun FaceDetectionOverlay(
    faces: List<FaceOverlayData>,
    modifier: Modifier = Modifier
) {
    Canvas(modifier = modifier) {
        faces.forEach { face ->
            val faceRect = mapRectToPreview(
                rect = face.boundingBox,
                imageWidth = face.imageWidth,
                imageHeight = face.imageHeight,
                previewWidth = size.width,
                previewHeight = size.height,
                mirrorHorizontally = true
            )
            val foreheadRect = mapRectToPreview(
                rect = face.foreheadRect,
                imageWidth = face.imageWidth,
                imageHeight = face.imageHeight,
                previewWidth = size.width,
                previewHeight = size.height,
                mirrorHorizontally = true
            )

            drawRect(
                color = Color(0xFF00E676),
                topLeft = Offset(faceRect.left, faceRect.top),
                size = Size(faceRect.width(), faceRect.height()),
                style = Stroke(width = 8.dp.toPx())
            )
            drawRect(
                color = Color(0xFFFF1744),
                topLeft = Offset(foreheadRect.left, foreheadRect.top),
                size = Size(foreheadRect.width(), foreheadRect.height()),
                style = Stroke(width = 6.dp.toPx())
            )

            val bracketWidth = faceRect.width() * 0.14f
            val bracketHeight = faceRect.height() * 0.14f
            val strokeWidth = 10.dp.toPx()

            drawBracket(faceRect.left, faceRect.top, bracketWidth, bracketHeight, strokeWidth, true, true)
            drawBracket(faceRect.right, faceRect.top, bracketWidth, bracketHeight, strokeWidth, false, true)
            drawBracket(faceRect.left, faceRect.bottom, bracketWidth, bracketHeight, strokeWidth, true, false)
            drawBracket(faceRect.right, faceRect.bottom, bracketWidth, bracketHeight, strokeWidth, false, false)
        }
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawBracket(
    x: Float,
    y: Float,
    bracketWidth: Float,
    bracketHeight: Float,
    strokeWidth: Float,
    leftSide: Boolean,
    topSide: Boolean
) {
    val horizontalEnd = if (leftSide) x + bracketWidth else x - bracketWidth
    val verticalEnd = if (topSide) y + bracketHeight else y - bracketHeight

    drawLine(
        color = Color(0xFF00E5FF),
        start = Offset(x, y),
        end = Offset(horizontalEnd, y),
        strokeWidth = strokeWidth,
        cap = StrokeCap.Round
    )
    drawLine(
        color = Color(0xFF00E5FF),
        start = Offset(x, y),
        end = Offset(x, verticalEnd),
        strokeWidth = strokeWidth,
        cap = StrokeCap.Round
    )
}

@Composable
internal fun OverlayChip(
    text: String,
    modifier: Modifier = Modifier,
    accent: Color = Color(0xFF00E5FF)
) {
    Text(
        text = text,
        modifier = modifier
            .clip(RoundedCornerShape(16.dp))
            .background(Color(0x99000000))
            .padding(horizontal = 16.dp, vertical = 10.dp),
        color = accent,
        style = MaterialTheme.typography.titleMedium
    )
}

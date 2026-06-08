package com.example.vpg

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

class FaceOverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var faceRect: RectF? = null
    private var roiRect: RectF? = null

    private val roiPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 8f
    }

    fun updateBoxes(face: RectF?, roi: RectF?) {
        this.faceRect = face
        this.roiRect = roi
        postInvalidate() // Wymusza odświeżenie ekranu
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        roiRect?.let { canvas.drawRect(it, roiPaint) }
    }
}
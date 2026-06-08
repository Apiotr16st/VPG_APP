package com.example.vpg

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker

class MainActivity : AppCompatActivity() {

    private lateinit var viewFinder: PreviewView
    private lateinit var overlayView: FaceOverlayView
    private lateinit var faceLandmarker: FaceLandmarker
    private lateinit var cameraExecutor: java.util.concurrent.ExecutorService

    private val roiAnalyzer = VpgRoiAnalyzer()


    private val heartRateAnalyzer = HeartRateAnalyzer()

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            setupMediaPipe()
            startCamera()
        } else {
            Toast.makeText(this, "Brak zgody na kamerę!", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewFinder = findViewById(R.id.viewFinder)
        overlayView = findViewById(R.id.overlayView) // Łapiemy "szybę" z XML

        cameraExecutor = java.util.concurrent.Executors.newSingleThreadExecutor()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            setupMediaPipe()
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun setupMediaPipe() {
        val baseOptions = BaseOptions.builder().setModelAssetPath("face_landmarker.task").build()
        val options = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.VIDEO)
            .setMinFaceDetectionConfidence(0.5f)
            .setMinFacePresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .build()

        faceLandmarker = FaceLandmarker.createFromOptions(this, options)
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(viewFinder.surfaceProvider)
            }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setTargetResolution(android.util.Size(480, 640))
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        processImage(imageProxy)
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
            } catch (exc: Exception) {
                Log.e("VPG", "Błąd uruchamiania kamery", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processImage(imageProxy: ImageProxy) {
        var bitmap: Bitmap? = null
        try {
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees
            val bitmapRaw = imageProxy.toBitmap()

            bitmap = if (rotationDegrees != 0) {
                val matrix = Matrix()
                matrix.postRotate(rotationDegrees.toFloat())
                val rotated = Bitmap.createBitmap(
                    bitmapRaw, 0, 0, bitmapRaw.width, bitmapRaw.height, matrix, true
                )
                bitmapRaw.recycle()
                rotated
            } else {
                bitmapRaw
            }

            val mpImage = BitmapImageBuilder(bitmap).build()
            val timestampMs = imageProxy.imageInfo.timestamp / 1000000

            val result = faceLandmarker.detectForVideo(mpImage, timestampMs)

            if (result.faceLandmarks().isNotEmpty()) {
                val vpgData = roiAnalyzer.processFrame(result, bitmap)

                if (vpgData != null) {
                    val screenRect = mapRectToScreen(
                        vpgData.roiRect,
                        bitmap.width,
                        bitmap.height,
                        overlayView.width,
                        overlayView.height
                    )

                    overlayView.updateBoxes(null, screenRect)

                    heartRateAnalyzer.processSamples(System.currentTimeMillis(), vpgData.meanY, vpgData.meanG)

                    val bpmG = heartRateAnalyzer.channelG.rollingBpmAverage

                    if (bpmG != null) {
                        Log.d("VPG_BPM", "❤️ TĘTNO (Kanał Zielony): ${"%.1f".format(bpmG)} BPM")
                    } else {
                        val zebrane = heartRateAnalyzer.channelG.samples.size
                        Log.d("VPG_BPM", "⏳ Zbieranie pulsu... ($zebrane / 150)")
                    }
                }
            } else {
                overlayView.updateBoxes(null, null)
                Log.d("VPG_SIGNAL", "Szukam twarzy...")
            }
        } catch (e: Exception) {
            Log.e("VPG", "Błąd analizy klatki", e)
        } finally {
            bitmap?.recycle()
            imageProxy.close() // BARDZO WAŻNE: zamykamy klatkę
        }
    }


    private fun mapRectToScreen(rect: android.graphics.Rect, imageW: Int, imageH: Int, viewW: Int, viewH: Int): android.graphics.RectF {
        val scaleX = viewW.toFloat() / imageW.toFloat()
        val scaleY = viewH.toFloat() / imageH.toFloat()
        val scale = Math.max(scaleX, scaleY)

        val scaledW = imageW * scale
        val scaledH = imageH * scale

        val dx = (viewW - scaledW) / 2f
        val dy = (viewH - scaledH) / 2f

        val scaledLeft = rect.left * scale + dx
        val scaledTop = rect.top * scale + dy
        val scaledRight = rect.right * scale + dx
        val scaledBottom = rect.bottom * scale + dy

        return android.graphics.RectF(
            viewW - scaledRight,
            scaledTop,
            viewW - scaledLeft,
            scaledBottom
        )
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}
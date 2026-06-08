package com.example.vpg

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.hardware.camera2.CaptureRequest
import android.os.Bundle
import android.util.Log
import android.util.Range
import android.util.Size
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
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
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private companion object {
        const val TARGET_CAMERA_FPS = 30
        val ANALYSIS_SIZE = Size(480, 640)
    }

    private lateinit var viewFinder: PreviewView
    private lateinit var overlayView: FaceOverlayView
    private lateinit var bpmText: TextView
    private lateinit var faceLandmarker: FaceLandmarker
    private lateinit var cameraExecutor: ExecutorService

    private val roiAnalyzer = VpgRoiAnalyzer()
    private val heartRateAnalyzer = HeartRateAnalyzer()

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { isGranted: Boolean ->
        if (isGranted) {
            setupMediaPipe()
            startCamera()
        } else {
            Toast.makeText(this, "Camera permission denied", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewFinder = findViewById(R.id.viewFinder)
        overlayView = findViewById(R.id.overlayView)
        bpmText = findViewById(R.id.bpmText)

        cameraExecutor = Executors.newSingleThreadExecutor()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            setupMediaPipe()
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun setupMediaPipe() {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("face_landmarker.task")
            .build()
        val options = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.VIDEO)
            .setMinFaceDetectionConfidence(0.5f)
            .setMinFacePresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .build()

        faceLandmarker = FaceLandmarker.createFromOptions(this, options)
    }

    @OptIn(ExperimentalCamera2Interop::class)
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val fpsRange = Range(TARGET_CAMERA_FPS, TARGET_CAMERA_FPS)
            val previewBuilder = Preview.Builder()
            Camera2Interop.Extender(previewBuilder).setCaptureRequestOption(
                CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE,
                fpsRange,
            )
            val preview = previewBuilder.build().also {
                it.setSurfaceProvider(viewFinder.surfaceProvider)
            }

            val imageAnalysisBuilder = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setTargetResolution(ANALYSIS_SIZE)
            Camera2Interop.Extender(imageAnalysisBuilder).setCaptureRequestOption(
                CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE,
                fpsRange,
            )
            val imageAnalyzer = imageAnalysisBuilder
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy: ImageProxy ->
                        processImage(imageProxy)
                    }
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_FRONT_CAMERA,
                    preview,
                    imageAnalyzer,
                )
            } catch (exc: Exception) {
                Log.e("VPG", "Failed to start camera", exc)
                runOnUiThread { bpmText.text = "BPM: camera unavailable" }
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
                    bitmapRaw,
                    0,
                    0,
                    bitmapRaw.width,
                    bitmapRaw.height,
                    matrix,
                    true,
                )
                bitmapRaw.recycle()
                rotated
            } else {
                bitmapRaw
            }

            val mpImage = BitmapImageBuilder(bitmap).build()
            val timestampMs = imageProxy.imageInfo.timestamp / 1_000_000
            val result = faceLandmarker.detectForVideo(mpImage, timestampMs)

            if (result.faceLandmarks().isNotEmpty()) {
                processFaceResult(result, bitmap, timestampMs)
            } else {
                overlayView.updateBoxes(null, null)
                runOnUiThread { bpmText.text = "BPM: face not detected" }
                Log.d("VPG_SIGNAL", "Face not detected")
            }
        } catch (e: Exception) {
            Log.e("VPG", "Frame analysis failed", e)
        } finally {
            bitmap?.recycle()
            imageProxy.close()
        }
    }

    private fun processFaceResult(
        result: com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult,
        bitmap: Bitmap,
        timestampMs: Long,
    ) {
        val vpgData = roiAnalyzer.processFrame(result, bitmap) ?: return
        val screenRect = mapRectToScreen(
            vpgData.roiRect,
            bitmap.width,
            bitmap.height,
            overlayView.width,
            overlayView.height,
        )

        overlayView.updateBoxes(null, screenRect)
        heartRateAnalyzer.processSamples(timestampMs, vpgData.meanY, vpgData.meanG)

        val channelG = heartRateAnalyzer.channelG
        val currentBpm = channelG.currentBpm
        val averageBpm = channelG.rollingBpmAverage
        if (currentBpm != null) {
            val averageText = averageBpm?.let { "%.1f".format(it) } ?: "--"
            val fpsText = channelG.lastBatchFps?.let { "%.1f".format(it) } ?: "--"
            val bpmTextValue = "BPM: ${"%.1f".format(currentBpm)} | avg10: $averageText | fps: $fpsText"
            runOnUiThread { bpmText.text = bpmTextValue }
            Log.d("VPG_BPM", "Green channel $bpmTextValue")
        } else {
            val collectedInBatch = channelG.samples.size - channelG.processedBatchSamples
            val progressText = "BPM: collecting signal ($collectedInBatch / ${heartRateAnalyzer.bufferSize})"
            runOnUiThread { bpmText.text = progressText }
            Log.d("VPG_BPM", progressText)
        }
    }

    private fun mapRectToScreen(
        rect: android.graphics.Rect,
        imageW: Int,
        imageH: Int,
        viewW: Int,
        viewH: Int,
    ): android.graphics.RectF {
        val scaleX = viewW.toFloat() / imageW.toFloat()
        val scaleY = viewH.toFloat() / imageH.toFloat()
        val scale = maxOf(scaleX, scaleY)

        val scaledW = imageW * scale
        val scaledH = imageH * scale
        val dx = (viewW - scaledW) / 2.0f
        val dy = (viewH - scaledH) / 2.0f

        val scaledLeft = rect.left * scale + dx
        val scaledTop = rect.top * scale + dy
        val scaledRight = rect.right * scale + dx
        val scaledBottom = rect.bottom * scale + dy

        return android.graphics.RectF(
            viewW - scaledRight,
            scaledTop,
            viewW - scaledLeft,
            scaledBottom,
        )
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}

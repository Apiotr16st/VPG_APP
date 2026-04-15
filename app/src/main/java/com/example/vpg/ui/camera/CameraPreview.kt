package com.example.vpg.ui.camera

import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.view.CameraController
import androidx.camera.view.LifecycleCameraController
import androidx.camera.view.PreviewView
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

@Composable
internal fun CameraPreview(
    modifier: Modifier = Modifier,
    onFacesDetected: (List<FaceOverlayData>) -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val latestOnFacesDetected = rememberUpdatedState(onFacesDetected)
    val analysisExecutor = remember { Executors.newSingleThreadExecutor() }
    val detector = remember {
        FaceDetection.getClient(
            FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                .build()
        )
    }
    val processingFrame = remember { AtomicBoolean(false) }
    val cameraController = remember(context) {
        LifecycleCameraController(context).apply {
            cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
            setEnabledUseCases(
                CameraController.IMAGE_CAPTURE or
                    CameraController.IMAGE_ANALYSIS or
                    CameraController.VIDEO_CAPTURE
            )
            imageAnalysisBackpressureStrategy = ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
        }
    }

    DisposableEffect(lifecycleOwner, cameraController, detector, analysisExecutor) {
        cameraController.setImageAnalysisAnalyzer(analysisExecutor) { imageProxy ->
            analyzeImageProxy(
                imageProxy = imageProxy,
                detector = detector,
                processingFrame = processingFrame,
                onFacesDetected = latestOnFacesDetected.value
            )
        }
        cameraController.bindToLifecycle(lifecycleOwner)

        onDispose {
            cameraController.clearImageAnalysisAnalyzer()
            cameraController.unbind()
            detector.close()
            analysisExecutor.shutdown()
        }
    }

    AndroidView(
        factory = { ctx ->
            PreviewView(ctx).apply {
                controller = cameraController
                implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                scaleType = PreviewView.ScaleType.FILL_CENTER
            }
        },
        modifier = modifier
    )
}

private fun analyzeImageProxy(
    imageProxy: ImageProxy,
    detector: FaceDetector,
    processingFrame: AtomicBoolean,
    onFacesDetected: (List<FaceOverlayData>) -> Unit
) {
    val mediaImage = imageProxy.image
    if (mediaImage == null) {
        imageProxy.close()
        return
    }

    if (!processingFrame.compareAndSet(false, true)) {
        imageProxy.close()
        return
    }

    val rotationDegrees = imageProxy.imageInfo.rotationDegrees
    val imageWidth = if (rotationDegrees == 90 || rotationDegrees == 270) {
        imageProxy.height
    } else {
        imageProxy.width
    }
    val imageHeight = if (rotationDegrees == 90 || rotationDegrees == 270) {
        imageProxy.width
    } else {
        imageProxy.height
    }
    val image = InputImage.fromMediaImage(mediaImage, rotationDegrees)

    detector.process(image)
        .addOnSuccessListener { faces ->
            handleDetectedFaces(faces, imageWidth, imageHeight, onFacesDetected)
        }
        .addOnFailureListener {
            FaceOverlayData.clearHistory()
            onFacesDetected(emptyList())
        }
        .addOnCompleteListener {
            processingFrame.set(false)
            imageProxy.close()
        }
}

private fun handleDetectedFaces(
    faces: List<Face>,
    imageWidth: Int,
    imageHeight: Int,
    onFacesDetected: (List<FaceOverlayData>) -> Unit
) {
    if (faces.isEmpty()) {
        FaceOverlayData.clearHistory()
        onFacesDetected(emptyList())
        return
    }

    onFacesDetected(
        faces.map { face ->
            face.toOverlayData(
                imageWidth = imageWidth,
                imageHeight = imageHeight
            )
        }
    )
}

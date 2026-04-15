package com.example.vpg.ui.camera

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.example.vpg.R
import com.example.vpg.ui.theme.VpgTheme

@Composable
fun FrontCameraScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCameraPermission = granted
    }

    LaunchedEffect(hasCameraPermission) {
        if (!hasCameraPermission) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    Box(
        modifier = modifier
            .fillMaxSize()
            .background(Color.Black),
        contentAlignment = Alignment.Center
    ) {
        if (hasCameraPermission) {
            FaceDetectionCameraScreen(modifier = Modifier.fillMaxSize())
        } else {
            PermissionFallback(
                onRequestPermission = {
                    permissionLauncher.launch(Manifest.permission.CAMERA)
                }
            )
        }
    }
}

@Composable
private fun PermissionFallback(onRequestPermission: () -> Unit) {
    Column(
        modifier = Modifier.padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = stringResource(R.string.camera_permission_message),
            color = Color.White,
            style = MaterialTheme.typography.bodyLarge
        )
        Button(onClick = onRequestPermission) {
            Text(text = stringResource(R.string.camera_permission_button))
        }
    }
}

@Composable
private fun FaceDetectionCameraScreen(modifier: Modifier = Modifier) {
    var faceOverlays by remember { mutableStateOf(emptyList<FaceOverlayData>()) }

    Box(
        modifier = modifier
            .fillMaxSize()
            .background(Color.Black)
    ) {
        CameraPreview(
            modifier = Modifier.fillMaxSize(),
            onFacesDetected = { faceOverlays = it }
        )
        FaceDetectionOverlay(
            modifier = Modifier.fillMaxSize(),
            faces = faceOverlays
        )
        if (faceOverlays.isEmpty()) {
            OverlayChip(
                text = stringResource(R.string.face_detector_waiting),
                accent = Color(0xFFFF7043),
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 24.dp)
            )
        }
    }
}

@Preview(showBackground = true)
@Composable
private fun FrontCameraScreenPreview() {
    VpgTheme {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black)
        ) {
            FaceDetectionOverlay(
                modifier = Modifier.fillMaxSize(),
                faces = listOf(
                    FaceOverlayData(
                        boundingBox = RectFCompat(220f, 260f, 860f, 1240f),
                        foreheadRect = RectFCompat(410f, 360f, 670f, 520f),
                        imageWidth = 1080f,
                        imageHeight = 1920f
                    )
                )
            )
        }
    }
}

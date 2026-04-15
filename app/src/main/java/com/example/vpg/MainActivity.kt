package com.example.vpg

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.ui.Modifier
import com.example.vpg.ui.camera.FrontCameraScreen
import com.example.vpg.ui.theme.VpgTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            VpgTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    FrontCameraScreen(
                        modifier = Modifier.padding(innerPadding)
                    )
                }
            }
        }
    }
}

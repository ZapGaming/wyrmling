package com.wyrmling.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import com.wyrmling.app.engine.MockEngine
import com.wyrmling.app.engine.WyrmEngine
import com.wyrmling.app.server.LocalApiServer
import com.wyrmling.app.ui.ChatScreen
import java.net.NetworkInterface

const val API_PORT = 8080

// Jade wyrmling + ember scale palette (distinct from Faultward crimson).
private val WyrmColors = darkColorScheme(
    primary = Color(0xFF34D399),
    onPrimary = Color(0xFF04140E),
    secondary = Color(0xFFF59E0B),
    background = Color(0xFF0E1512),
    onBackground = Color(0xFFE8F0EC),
    surface = Color(0xFF14201B),
    onSurface = Color(0xFFE8F0EC),
    surfaceVariant = Color(0xFF1C2A24),
    outline = Color(0xFF2E4038),
)

class MainActivity : ComponentActivity() {
    private lateinit var engine: WyrmEngine
    private var server: LocalApiServer? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        engine = MockEngine()  // swap for LlamaCppEngine(ggufPath) when the model is trained + converted
        server = LocalApiServer(engine, API_PORT).also {
            runCatching { it.start(NanoTimeout, false) }
        }
        val addr = "http://${lanIp() ?: "127.0.0.1"}:$API_PORT/v1"
        setContent {
            MaterialTheme(colorScheme = WyrmColors) {
                ChatScreen(engine = engine, apiAddress = addr)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        server?.stop()
    }

    private companion object { const val NanoTimeout = 5000 }
}

private fun lanIp(): String? = runCatching {
    NetworkInterface.getNetworkInterfaces().toList()
        .flatMap { it.inetAddresses.toList() }
        .firstOrNull { !it.isLoopbackAddress && it.hostAddress?.contains(':') == false }
        ?.hostAddress
}.getOrNull()

@Composable
private fun themePreviewAnchor() { /* keeps ui-tooling happy */ }

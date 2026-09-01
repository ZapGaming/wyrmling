package com.wyrmling.app.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.runtime.toMutableStateList
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.wyrmling.app.engine.WyrmEngine
import kotlinx.coroutines.launch

private class Msg(role: String, text: String) {
    val role = role
    var text by mutableStateOf(text)
}

@Composable
fun ChatScreen(engine: WyrmEngine, apiAddress: String) {
    val scope = rememberCoroutineScope()
    val messages = remember {
        listOf(Msg("assistant", "🐉 Wyrmling is awake. Say something — and any OpenAI client can reach me at the address above.")).toMutableStateList()
    }
    var input by remember { mutableStateOf("") }
    var busy by remember { mutableStateOf(false) }
    val listState = rememberLazyListState()

    fun send() {
        val t = input.trim()
        if (t.isEmpty() || busy) return
        input = ""
        busy = true
        messages.add(Msg("user", t))
        val reply = Msg("assistant", "")
        messages.add(reply)
        scope.launch {
            val convo = messages.filter { it.text.isNotEmpty() }
                .joinToString("\n") { "${it.role}: ${it.text}" }
            engine.generate(convo, onToken = { reply.text += it })
            busy = false
            listState.animateScrollToItem(messages.lastIndex)
        }
    }

    Scaffold(
        containerColor = MaterialTheme.colorScheme.background,
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("Wyrmling 🐉", fontWeight = FontWeight.Bold)
                        Text(
                            apiAddress,
                            fontSize = 11.sp,
                            color = MaterialTheme.colorScheme.primary,
                        )
                    }
                },
                actions = { StatusChip(engine.ready, engine.displayName) },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface,
                    titleContentColor = MaterialTheme.colorScheme.onSurface,
                ),
            )
        },
        bottomBar = { InputBar(input, { input = it }, ::send, busy) },
    ) { pad ->
        LazyColumn(
            state = listState,
            modifier = Modifier.fillMaxSize().padding(pad).padding(horizontal = 12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            item { Box(Modifier.padding(4.dp)) }
            items(messages) { Bubble(it) }
        }
    }
}

@Composable
private fun StatusChip(ready: Boolean, label: String) {
    val c = if (ready) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.secondary
    Row(
        modifier = Modifier
            .padding(end = 12.dp)
            .background(c.copy(alpha = 0.15f), RoundedCornerShape(50))
            .padding(horizontal = 10.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(Modifier.background(c, RoundedCornerShape(50)).padding(4.dp))
        Text("  " + (if (ready) "live" else "demo"), color = c, fontSize = 11.sp)
    }
}

@Composable
private fun Bubble(m: Msg) {
    val mine = m.role == "user"
    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = if (mine) Arrangement.End else Arrangement.Start,
    ) {
        Surface(
            color = if (mine) MaterialTheme.colorScheme.primary
            else MaterialTheme.colorScheme.surfaceVariant,
            contentColor = if (mine) MaterialTheme.colorScheme.onPrimary
            else MaterialTheme.colorScheme.onBackground,
            shape = RoundedCornerShape(16.dp),
            modifier = Modifier.widthIn(max = 300.dp),
        ) {
            Text(
                m.text.ifEmpty { "…" },
                modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
                fontSize = 15.sp,
            )
        }
    }
}

@Composable
private fun InputBar(value: String, onChange: (String) -> Unit, onSend: () -> Unit, busy: Boolean) {
    Surface(color = MaterialTheme.colorScheme.surface) {
        Row(
            Modifier.fillMaxWidth().padding(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            TextField(
                value = value,
                onValueChange = onChange,
                modifier = Modifier.weight(1f),
                placeholder = { Text("Message Wyrmling…") },
                maxLines = 4,
                keyboardActions = KeyboardActions(onSend = { onSend() }),
                colors = TextFieldDefaults.colors(
                    focusedContainerColor = MaterialTheme.colorScheme.surfaceVariant,
                    unfocusedContainerColor = MaterialTheme.colorScheme.surfaceVariant,
                    focusedIndicatorColor = Color.Transparent,
                    unfocusedIndicatorColor = Color.Transparent,
                ),
                shape = RoundedCornerShape(24.dp),
            )
            IconButton(onClick = onSend, enabled = !busy) {
                Icon(
                    Icons.AutoMirrored.Filled.Send,
                    contentDescription = "Send",
                    tint = if (busy) MaterialTheme.colorScheme.outline
                    else MaterialTheme.colorScheme.primary,
                )
            }
        }
    }
}

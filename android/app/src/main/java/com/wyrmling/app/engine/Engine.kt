package com.wyrmling.app.engine

import kotlinx.coroutines.delay

/**
 * The inference seam. The app + local API talk to this, never to a specific backend.
 * Swap [MockEngine] for [ExecuTorchEngine] (or a llama.cpp/GGUF JNI engine) once a
 * trained Wyrmling model + on-device runtime exist — nothing else changes.
 */
interface WyrmEngine {
    /** Human label shown in the UI status chip. */
    val displayName: String
    /** True when a real model is loaded and can answer. */
    val ready: Boolean

    /**
     * Generate a reply to [prompt] (already the full conversation text).
     * Emits decoded pieces via [onToken] as they arrive; returns the full text.
     */
    suspend fun generate(
        prompt: String,
        maxTokens: Int = 256,
        temperature: Float = 0.8f,
        onToken: (String) -> Unit = {},
    ): String
}

/**
 * Demo backend — no model. Streams a canned, playful Wyrmling reply so the UI and the
 * local OpenAI API are fully exercisable before any weights exist. Clearly labelled so
 * nobody mistakes it for the trained model.
 */
class MockEngine : WyrmEngine {
    override val displayName = "Demo · no model loaded"
    override val ready = true

    override suspend fun generate(
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        onToken: (String) -> Unit,
    ): String {
        val last = prompt.substringAfterLast("\n").take(120).ifBlank { "…" }
        val reply = "🐉 (demo) A wyrmling stirs. You said: \"$last\". " +
            "I'm the UI + local OpenAI API running end-to-end — the real byte-level BDH " +
            "brain drops in here once it's trained and ported on-device."
        val sb = StringBuilder()
        for (word in reply.split(" ")) {
            val piece = if (sb.isEmpty()) word else " $word"
            sb.append(piece)
            onToken(piece)
            delay(28)
        }
        return sb.toString()
    }
}

/**
 * Real on-device backend — STUB. Wyrmling is Mamba-2 → GGUF-native, so this wires to a
 * **llama.cpp GGUF runtime** (JNI / llama.android): load the `.gguf`, run, stream bytes.
 * Seamless across platforms — no per-device port. See ../../../../GGUF.md + MOBILE.md.
 */
class LlamaCppEngine(private val ggufPath: String) : WyrmEngine {
    override val displayName = "llama.cpp GGUF (model not loaded)"
    override val ready = false

    override suspend fun generate(
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        onToken: (String) -> Unit,
    ): String {
        // TODO: JNI into llama.cpp, load wyrmling-*.gguf, decode, stream bytes to onToken.
        val msg = "[on-device model not yet wired — train Wyrmling, convert to .gguf, drop it here]"
        onToken(msg)
        return msg
    }
}

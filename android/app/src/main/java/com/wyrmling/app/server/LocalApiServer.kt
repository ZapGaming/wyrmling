package com.wyrmling.app.server

import com.wyrmling.app.engine.WyrmEngine
import fi.iki.elonen.NanoHTTPD
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject

/**
 * Embedded OpenAI-compatible HTTP server, on-device. Point any OpenAI SDK at
 * http://<phone-ip>:<port>/v1. Endpoints: GET /health, GET /v1/models,
 * POST /v1/chat/completions (non-streaming JSON). Backed by the pluggable [engine].
 */
class LocalApiServer(
    private val engine: WyrmEngine,
    port: Int = 8080,
) : NanoHTTPD("0.0.0.0", port) {

    val modelId get() = "wyrmling"

    override fun serve(session: IHTTPSession): Response {
        return try {
            when {
                session.uri == "/health" -> json(JSONObject().put("status", "ok").put("model", modelId))
                session.uri.trimEnd('/') == "/v1/models" -> models()
                session.method == Method.POST && session.uri.trimEnd('/') == "/v1/chat/completions" ->
                    chat(session)
                else -> json(JSONObject().put("error", "not found"), Response.Status.NOT_FOUND)
            }
        } catch (e: Exception) {
            json(JSONObject().put("error", e.message ?: "server error"),
                Response.Status.INTERNAL_ERROR)
        }
    }

    private fun models(): Response {
        val data = JSONArray().put(
            JSONObject().put("id", modelId).put("object", "model").put("owned_by", "wyrmling")
        )
        return json(JSONObject().put("object", "list").put("data", data))
    }

    private fun chat(session: IHTTPSession): Response {
        val body = HashMap<String, String>()
        session.parseBody(body)
        val req = JSONObject(body["postData"] ?: "{}")
        val messages = req.optJSONArray("messages") ?: JSONArray()
        val maxTokens = req.optInt("max_tokens", 256)
        val temp = req.optDouble("temperature", 0.8).toFloat()

        val sb = StringBuilder()
        for (i in 0 until messages.length()) {
            val m = messages.getJSONObject(i)
            sb.append(m.optString("role")).append(": ").append(m.optString("content")).append("\n")
        }
        val text = runBlocking { engine.generate(sb.toString(), maxTokens, temp) }

        val created = System.currentTimeMillis() / 1000
        val choice = JSONObject()
            .put("index", 0)
            .put("message", JSONObject().put("role", "assistant").put("content", text))
            .put("finish_reason", "stop")
        val resp = JSONObject()
            .put("id", "chatcmpl-$created").put("object", "chat.completion")
            .put("created", created).put("model", modelId)
            .put("choices", JSONArray().put(choice))
            .put("usage", JSONObject()
                .put("prompt_tokens", 0).put("completion_tokens", text.length)
                .put("total_tokens", text.length))
        return json(resp)
    }

    private fun json(obj: JSONObject, status: Response.Status = Response.Status.OK): Response {
        val r = newFixedLengthResponse(status, "application/json", obj.toString())
        r.addHeader("Access-Control-Allow-Origin", "*")
        return r
    }
}

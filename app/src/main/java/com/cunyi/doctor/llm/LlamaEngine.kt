package com.cunyi.doctor.llm

import android.content.Context
import android.util.Log
import com.cunyi.doctor.data.ModelDownloadManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withContext
import java.io.File

class LlamaEngine private constructor(private val context: Context) {

    companion object {
        private const val TAG = "LlamaEngine"
        private var INSTANCE: LlamaEngine? = null

        init {
            System.loadLibrary("cunyi_doctor")
        }

        fun getInstance(ctx: Context): LlamaEngine {
            return INSTANCE ?: LlamaEngine(ctx.applicationContext).also { INSTANCE = it }
        }

        // ── 可选模型列表 ──────────────────────────────────────────────────
        val MODELS = listOf(
            ModelConfig(
                id = "qwen2.5-0.5b",
                name = "Qwen2.5-0.5B",
                desc = "轻量通用模型，速度快，约 500MB",
                url = "https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
                file = "qwen2.5-0.5b-instruct-q4_k_m.gguf",
                size = "500 MB",
                promptFormat = ModelPromptFormat.QWEN
            ),
            ModelConfig(
                id = "medgemma-1.5-4b",
                name = "MedGemma-1.5-4B",
                desc = "医疗专精模型，效果更好，约 2.5GB",
                url = "https://www.modelscope.cn/models/unsloth/medgemma-1.5-4b-it-GGUF/resolve/master/medgemma-1.5-4b-it-Q4_K_M.gguf",
                file = "medgemma-1.5-4b-it-q4_k_m.gguf",
                size = "2.5 GB",
                promptFormat = ModelPromptFormat.GEMMA
            )
        )
    }

    private var isLoaded = false
    private var isGenerating = false
    private var selectedModel: ModelConfig = MODELS[0]  // 默认 Qwen

    // Expose state
    val modelLoadState = MutableStateFlow<LoadState>(LoadState.Idle)
    val generationState = MutableStateFlow<GenState>(GenState.Idle)

    val isModelLoaded: Boolean get() = isLoaded
    val isRunning: Boolean get() = isGenerating

    // ── 模型选择 ──────────────────────────────────────────────────────────
    fun selectModel(model: ModelConfig) {
        if (isLoaded) {
            Log.w(TAG, "Cannot change model while loaded")
            return
        }
        selectedModel = model
    }

    fun getSelectedModel(): ModelConfig = selectedModel

    fun getModelFile(): File = File(context.filesDir, selectedModel.file)

    /** 扫描已下载的模型文件，返回其 ModelConfig */
    fun findDownloadedModel(): ModelConfig? {
        return MODELS.firstOrNull { File(context.filesDir, it.file).exists() }
    }

    suspend fun loadModel(progress: (Int, String) -> Unit = { _, _ -> }): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            val file = getModelFile()
            Log.i(TAG, "getModelFile: ${file.absolutePath}")

            if (!file.exists()) {
                progress(0, "正在下载 ${selectedModel.name}...")
                val downloadResult = ModelDownloadManager.download(
                    context = context,
                    url = selectedModel.url,
                    fileName = selectedModel.file
                ) { dlProgress, dlMsg ->
                    progress(dlProgress, dlMsg)
                }
                if (downloadResult.isFailure) {
                    val err = downloadResult.exceptionOrNull()?.message ?: "下载失败"
                    Log.e(TAG, "Download failed: $err")
                    return@withContext Result.failure(Exception(err))
                }
                Log.i(TAG, "Download complete: ${file.absolutePath}, size=${file.length()}")
            } else {
                Log.i(TAG, "Model file exists: ${file.absolutePath}, size=${file.length()}")
                progress(90, "模型文件已存在，开始加载...")
            }

            // 验证文件完整性
            if (!file.exists()) {
                return@withContext Result.failure(Exception("模型文件不存在"))
            }
            val fileSize = file.length()
            if (fileSize < 100_000_000) {  // 至少 100MB
                return@withContext Result.failure(Exception("模型文件太小 ($fileSize bytes)，可能下载不完整"))
            }
            
            progress(95, "正在加载模型 (${fileSize / 1024 / 1024} MB)...")
            Log.i(TAG, "Calling native _init with path: ${file.absolutePath}")
            
            val success = _init(file.absolutePath)
            Log.i(TAG, "_init returned: $success")
            
            if (success) {
                isLoaded = true
                modelLoadState.value = LoadState.Loaded
                progress(100, "模型加载完成")
                Log.i(TAG, "Model loaded successfully")
                Result.success(Unit)
            } else {
                val err = _getLastError() ?: "未知错误"
                Log.e(TAG, "_init failed: $err")
                modelLoadState.value = LoadState.Error(err)
                Result.failure(Exception(err))
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Load error", t)
            modelLoadState.value = LoadState.Error(t.message ?: "加载失败")
            Result.failure(if (t is Exception) t else Exception(t))
        }
    }

    // ── Native methods ──────────────────────────────────────────────────────
    private external fun _init(modelPath: String): Boolean
    private external fun _isLoaded(): Boolean
    private external fun _getLastError(): String?
    private external fun _generateAsync(prompt: String, callback: GenerationCallback)
    private external fun _stop()
    private external fun _isRunning(): Boolean
    private external fun _release()

    // ── Generation ───────────────────────────────────────────────────────────
    fun generate(
        prompt: String,
        onToken: (String) -> Unit,
        onDone: (String) -> Unit,
        onError: (String) -> Unit
    ) {
        if (!isLoaded) {
            onError("模型未加载")
            return
        }
        if (isGenerating) {
            onError("正在生成中，请稍候")
            return
        }
        isGenerating = true
        generationState.value = GenState.Generating

        val fullPrompt = buildPrompt(prompt)
        _generateAsync(fullPrompt, object : GenerationCallback {
            override fun onToken(token: String) {
                onToken(token)
            }
            override fun onDone(fullText: String) {
                isGenerating = false
                generationState.value = GenState.Idle
                onDone(fullText)
            }
            override fun onError(error: String) {
                isGenerating = false
                generationState.value = GenState.Error(error)
                onError(error)
            }
        })
    }

    fun stop() {
        if (isGenerating) {
            _stop()
            isGenerating = false
            generationState.value = GenState.Idle
        }
    }

    private fun buildPrompt(userMessage: String): String {
        return when (selectedModel.promptFormat) {
            ModelPromptFormat.QWEN -> {
                """<|im_start|>system
你是一位专业、温暖、有耐心的村医。请根据以下问题给出详细、易懂的健康建议。<|im_end|>
<|im_start|>user
$userMessage<|im_end|>
<|im_start|>assistant
"""
            }
            ModelPromptFormat.GEMMA -> {
                // Gemma / MedGemma prompt 格式
                "<start_of_turn>user\n你是一位专业、温暖、有耐心的村医。请根据以下问题给出详细、易懂的健康建议。\n\n$userMessage<end_of_turn>\n<start_of_turn>model\n"
            }
        }
    }

    fun release() {
        _release()
        isLoaded = false
        isGenerating = false
        modelLoadState.value = LoadState.Idle
        generationState.value = GenState.Idle
    }

    // State classes
    sealed class LoadState {
        data object Idle : LoadState()
        data object Loading : LoadState()
        data object Loaded : LoadState()
        data class Error(val msg: String) : LoadState()
    }

    sealed class GenState {
        data object Idle : GenState()
        data object Generating : GenState()
        data class Error(val msg: String) : GenState()
    }

    interface GenerationCallback {
        fun onToken(token: String)
        fun onDone(fullText: String)
        fun onError(error: String)
    }
}

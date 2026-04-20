package com.xueba.emperor.llm

import android.content.Context
import android.util.Log
import com.xueba.emperor.data.ModelDownloadManager
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
            System.loadLibrary("xueba")
        }

        fun getInstance(ctx: Context): LlamaEngine {
            return INSTANCE ?: LlamaEngine(ctx.applicationContext).also { INSTANCE = it }
        }

        // ── 模型列表 ──────────────────────────────────────────────────────
        val MODELS = listOf(
            ModelConfig(
                id = "qwen2.5-0.5b",
                name = "Qwen2.5-0.5B",
                desc = "通用学霸模型，离线运行，约 500MB",
                url = "https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/master/qwen2.5-0.5b-instruct-q4_k_m.gguf",
                file = "qwen2.5-0.5b-instruct-q4_k_m.gguf",
                size = "500 MB",
                promptFormat = ModelPromptFormat.QWEN
            ),
            ModelConfig(
                id = "gemma-4-e2b",
                name = "Gemma-4-E2B",
                desc = "Google 高性能模型，离线运行，约 1.5GB",
                url = "https://www.modelscope.cn/models/unsloth/gemma-4-E2B-it-GGUF/resolve/master/gemma-4-E2B-it-Q4_K_M.gguf",
                backupUrl = "https://www.modelscope.cn/models/unsloth/gemma-4-E2B-it-GGUF/resolve/master/gemma-4-E2B-it-Q3_K_M.gguf",
                file = "gemma-4-E2B-it-Q4_K_M.gguf",
                size = "1.5 GB",
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
                var downloadResult = ModelDownloadManager.download(
                    context = context,
                    url = selectedModel.url,
                    fileName = selectedModel.file
                ) { dlProgress, dlMsg ->
                    progress(dlProgress, dlMsg)
                }
                // 主地址失败 → 尝试备用地址
                if (downloadResult.isFailure && selectedModel.backupUrl.isNotEmpty()) {
                    Log.w(TAG, "Primary download failed, trying backup URL...")
                    progress(0, "主下载地址失败，尝试备用地址...")
                    downloadResult = ModelDownloadManager.download(
                        context = context,
                        url = selectedModel.backupUrl,
                        fileName = selectedModel.file
                    ) { dlProgress, dlMsg ->
                        progress(dlProgress, dlMsg)
                    }
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
你是一个学识渊博、逻辑严谨、表达清晰的学霸。无论对方问什么领域的问题——数学、物理、化学、生物、历史、地理、语文、英语、编程、学习方法、竞赛题目——你都能给出准确、详尽、深入浅出的解答。你的风格是：严谨但不刻板，博学但不卖弄，总能用最清晰的方式把复杂的问题讲明白。遇到难题你会一步步拆解，遇到计算题你会给出完整推导过程。必要时给出例题和方法论，让对方不仅知道答案，更理解背后的原理。<|im_end|>
<|im_start|>user
$userMessage<|im_end|>
<|im_start|>assistant
"""
            }
            ModelPromptFormat.GEMMA -> {
                "<start_of_turn>user\n你是一个学识渊博、逻辑严谨、表达清晰的学霸。无论对方问什么领域的问题——数学、物理、化学、生物、历史、地理、语文、英语、编程、学习方法、竞赛题目——你都能给出准确、详尽、深入浅出的解答。遇到难题你会一步步拆解，遇到计算题你会给出完整推导过程。\n\n$userMessage<end_of_turn>\n<start_of_turn>model\n"
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

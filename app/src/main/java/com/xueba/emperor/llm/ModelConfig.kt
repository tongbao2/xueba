package com.xueba.emperor.llm

/**
 * 模型配置（top-level class，供其他文件引用）
 */
data class ModelConfig(
    val id: String,
    val name: String,
    val desc: String,
    val url: String,
    val backupUrl: String = "",
    val file: String,
    val size: String,
    val promptFormat: ModelPromptFormat
)

enum class ModelPromptFormat { QWEN, GEMMA, DEEPSEEK_R1 }

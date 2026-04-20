# 🦞学霸帝🦞

基于 **llama.cpp + GGUF** 的本地离线 Android 应用，无需网络即可运行大语言模型推理。

## ✨ 特性

- 🔒 **完全离线** — 模型下载后无需网络，数据不上传
- 🧠 **本地推理** — 基于 llama.cpp JNI 桥接，纯 CPU 运行
- 🗣️ **语音播报** — TTS 自动朗读 AI 回复
- 📸 **图片 OCR** — 拍照/选图识别文字，结合 LLM 解答
- 💬 **流式输出** — 逐 token 生成，实时显示回复内容
- 📱 **多模型选择** — 支持 Qwen2.5 和 Gemma 系列

## 🏗️ 技术架构

```
┌─────────────────────────────────┐
│         Android UI Layer        │
│  MainActivity / MainViewModel   │
│    ChatAdapter / TTS / OCR      │
├─────────────────────────────────┤
│        Kotlin Bridge Layer      │
│  LlamaEngine / ModelDownload    │
├─────────────────────────────────┤
│          JNI Native Layer       │
│  llama_jni.cpp → llama.cpp     │
│     (CMake + NDK cross-compile) │
└─────────────────────────────────┘
```

## 🔧 编译环境

- Android SDK 35 (Build Tools 35.0.0)
- NDK 25.1.8937393
- CMake 3.22.1
- Java 17
- Gradle 8.5

## 📦 构建步骤

```bash
# 1. 克隆仓库（含 submodule）
git clone --recurse-submodules https://github.com/tongbao2/xuebadi.git
cd xuebadi

# 2. 编译 Debug APK
./gradlew assembleDebug

# 3. APK 位置
# app/build/outputs/apk/debug/app-debug.apk
```

## 📲 使用方法

1. 安装 APK 到 Android 设备
2. 首次打开点击「📥 下载模型」按钮
3. 选择模型后等待下载完成
4. 开始对话

## 🤖 支持的模型

| 模型 | 大小 | 说明 |
|------|------|------|
| Qwen2.5-0.5B-Instruct-Q4_K_M | ~500MB | 默认，速度快 |
| gemma-4-E2B-it-Q4_K_M | ~1.5GB | Google 大模型，效果更好 |

模型来源：[ModelScope](https://www.modelscope.cn)

## 📁 项目结构

```
app/src/main/
├── cpp/
│   ├── CMakeLists.txt
│   ├── llama_jni.cpp          # JNI 桥接层
│   └── llama.cpp-src/         # llama.cpp (git submodule)
├── java/com/xueba/emperor/
│   ├── XuebaEmperorApp.kt     # Application
│   ├── data/
│   │   └── ModelDownloadManager.kt
│   ├── llm/
│   │   ├── LlamaEngine.kt     # 模型加载/推理引擎
│   │   └── ModelConfig.kt     # 模型配置
│   ├── ui/
│   │   ├── MainActivity.kt    # 主界面
│   │   ├── MainViewModel.kt   # ViewModel
│   │   └── ChatAdapter.kt     # 聊天列表适配器
│   └── utils/
│       └── ImageTextRecognizer.kt  # ML Kit OCR
└── res/
    ├── layout/
    ├── drawable/
    └── values/
```

## ⚠️ 注意事项

- 纯 CPU 推理，大模型加载和生成速度较慢属正常现象
- 首次加载模型需 30秒-2分钟，请耐心等待
- 模型文件约 500MB~1.5GB，请确保存储空间充足
- 建议在 Wi-Fi 环境下下载模型

## 📄 License

MIT

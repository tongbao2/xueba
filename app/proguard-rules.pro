# Keep llama JNI classes
-keep class com.xueba.emperor.llm.** { *; }
-keepclasseswithmembernames class * {
    native <methods>;
}

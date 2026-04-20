package com.xueba.emperor

import android.app.Application

class XuebaEmperorApp : Application() {
    override fun onCreate() {
        super.onCreate()
        instance = this
    }

    companion object {
        lateinit var instance: XuebaEmperorApp
            private set
    }
}

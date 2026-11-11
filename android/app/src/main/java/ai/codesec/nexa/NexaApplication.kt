package ai.codesec.nexa

import android.app.Application
import android.content.ComponentCallbacks2
import android.util.Log
import ai.codesec.mobilellm.MobileLLMClient

/**
 * NexaApplication manages application-level lifecycle and coordinates
 * OS memory pressure events with the native MobileLLM runtime.
 */
class NexaApplication : Application() {

    companion object {
        private const val TAG = "NexaApplication"
        lateinit var instance: NexaApplication
            private set
    }

    val mobileLLMClient: MobileLLMClient by lazy {
        MobileLLMClient.getInstance(this)
    }

    override fun onCreate() {
        super.onCreate()
        instance = this
        Log.i(TAG, "NEXA Offline LLM Assistant initialized.")
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        Log.w(TAG, "onTrimMemory received: level=$level")
        when (level) {
            ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL,
            ComponentCallbacks2.TRIM_MEMORY_COMPLETE -> {
                Log.w(TAG, "Critical memory pressure. Purging session KV caches.")
                mobileLLMClient.onMemoryPressure(level)
            }
            ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW,
            ComponentCallbacks2.TRIM_MEMORY_MODERATE -> {
                Log.w(TAG, "Moderate memory pressure. Trimming idle caches.")
                mobileLLMClient.onMemoryPressure(level)
            }
        }
    }

    override fun onLowMemory() {
        super.onLowMemory()
        Log.e(TAG, "System in critically low memory condition!")
        mobileLLMClient.onMemoryPressure(ComponentCallbacks2.TRIM_MEMORY_COMPLETE)
    }
}


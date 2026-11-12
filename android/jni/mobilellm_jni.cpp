#include <jni.h>
#include "mobilellm/c_api.h"
#include <string>
#include <memory>
#include <cstring>

namespace {

JavaVM* g_jvm = nullptr;

class JStringGuard {
public:
    JStringGuard(JNIEnv* env, jstring jstr) : env_(env), jstr_(jstr) {
        if (jstr_) {
            chars_ = env_->GetStringUTFChars(jstr_, nullptr);
        }
    }

    ~JStringGuard() {
        if (jstr_ && chars_) {
            env_->ReleaseStringUTFChars(jstr_, chars_);
        }
    }

    const char* c_str() const {
        return chars_ ? chars_ : "";
    }

private:
    JNIEnv* env_;
    jstring jstr_;
    const char* chars_{nullptr};
};

void throw_java_exception(JNIEnv* env, int32_t code, const char* message) {
    jclass ex_class = env->FindClass("ai/codesec/mobilellm/MobileLLMException");
    if (!ex_class) {
        // Fallback to standard RuntimeException if custom class not found
        ex_class = env->FindClass("java/lang/RuntimeException");
    }
    if (ex_class) {
        std::string err = "[" + std::to_string(code) + "] " + (message ? message : "Native error");
        env->ThrowNew(ex_class, err.c_str());
    }
}

struct JavaCallbackContext {
    jobject global_callback{nullptr};

    ~JavaCallbackContext() {
        if (global_callback && g_jvm) {
            JNIEnv* env = nullptr;
            bool attached = false;
            jint res = g_jvm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
            if (res == JNI_EDETACHED) {
                if (g_jvm->AttachCurrentThread(&env, nullptr) == JNI_OK) {
                    attached = true;
                }
            }
            if (env) {
                env->DeleteGlobalRef(global_callback);
                global_callback = nullptr;
            }
            if (attached) {
                g_jvm->DetachCurrentThread();
            }
        }
    }
};

} // namespace

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* /*reserved*/) {
    g_jvm = vm;
    return JNI_VERSION_1_6;
}

extern "C" {

JNIEXPORT jlong JNICALL
Java_ai_codesec_mobilellm_NativeBridge_nativeInit(JNIEnv* env, jclass /*clazz*/) {
    mllm_engine_t engine = 0;
    mllm_status_t st = mllm_engine_create(&engine);
    if (st.code != 0) {
        throw_java_exception(env, st.code, st.message);
        return 0;
    }
    return static_cast<jlong>(engine);
}

JNIEXPORT void JNICALL
Java_ai_codesec_mobilellm_NativeBridge_nativeDestroy(
    JNIEnv* env,
    jclass /*clazz*/,
    jlong engine_handle
) {
    if (engine_handle == 0) return;
    mllm_status_t st = mllm_engine_destroy(static_cast<mllm_engine_t>(engine_handle));
    if (st.code != 0) {
        throw_java_exception(env, st.code, st.message);
    }
}

JNIEXPORT jlong JNICALL
Java_ai_codesec_mobilellm_NativeBridge_nativeLoadModel(
    JNIEnv* env,
    jclass /*clazz*/,
    jlong engine_handle,
    jstring model_path,
    jint backend,
    jint context_length,
    jint n_threads,
    jboolean use_mmap,
    jfloat memory_budget_mb
) {
    if (!model_path) {
        throw_java_exception(env, 1, "model_path cannot be null");
        return 0;
    }

    JStringGuard path_guard(env, model_path);

    mllm_model_config_t cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.model_path = path_guard.c_str();
    cfg.backend = backend;
    cfg.context_length = context_length > 0 ? context_length : 2048;
    cfg.n_threads = n_threads > 0 ? n_threads : 4;
    cfg.use_mmap = use_mmap;
    cfg.memory_budget_mb = memory_budget_mb;

    mllm_model_t model = 0;
    mllm_status_t st = mllm_model_load(static_cast<mllm_engine_t>(engine_handle), &cfg, &model);
    if (st.code != 0) {
        throw_java_exception(env, st.code, st.message);
        return 0;
    }
    return static_cast<jlong>(model);
}

JNIEXPORT void JNICALL
Java_ai_codesec_mobilellm_NativeBridge_nativeUnloadModel(
    JNIEnv* env,
    jclass /*clazz*/,
    jlong engine_handle,
    jlong model_handle
) {
    mllm_status_t st = mllm_model_unload(
        static_cast<mllm_engine_t>(engine_handle),
        static_cast<mllm_model_t>(model_handle)
    );
    if (st.code != 0) {
        throw_java_exception(env, st.code, st.message);
    }
}

JNIEXPORT jlong JNICALL
Java_ai_codesec_mobilellm_NativeBridge_nativeCreateSession(
    JNIEnv* env,
    jclass /*clazz*/,
    jlong engine_handle,
    jlong model_handle,
    jint max_context_tokens,
    jboolean enable_prefix_caching
) {
    mllm_session_config_t cfg;
    cfg.max_context_tokens = max_context_tokens > 0 ? max_context_tokens : 2048;
    cfg.enable_prefix_caching = enable_prefix_caching;

    mllm_session_t session = 0;
    mllm_status_t st = mllm_session_create(
        static_cast<mllm_engine_t>(engine_handle),
        static_cast<mllm_model_t>(model_handle),
        &cfg,
        &session
    );

    if (st.code != 0) {
        throw_java_exception(env, st.code, st.message);
        return 0;
    }
    return static_cast<jlong>(session);
}

JNIEXPORT void JNICALL
Java_ai_codesec_mobilellm_NativeBridge_nativeDestroySession(
    JNIEnv* env,
    jclass /*clazz*/,
    jlong engine_handle,
    jlong session_handle
) {
    mllm_status_t st = mllm_session_destroy(
        static_cast<mllm_engine_t>(engine_handle),
        static_cast<mllm_session_t>(session_handle)
    );
    if (st.code != 0) {
        throw_java_exception(env, st.code, st.message);
    }
}

JNIEXPORT jlong JNICALL
Java_ai_codesec_mobilellm_NativeBridge_nativeGenerate(
    JNIEnv* env,
    jclass /*clazz*/,
    jlong engine_handle,
    jlong session_handle,
    jstring prompt,
    jfloat temperature,
    jfloat top_p,
    jint top_k,
    jfloat min_p,
    jfloat rep_penalty,
    jint max_tokens,
    jobject callback
) {
    if (!prompt) {
        throw_java_exception(env, 1, "prompt cannot be null");
        return 0;
    }

    JStringGuard prompt_guard(env, prompt);

    mllm_sampling_params_t sparams;
    std::memset(&sparams, 0, sizeof(sparams));
    sparams.temperature = temperature;
    sparams.top_p = top_p;
    sparams.top_k = top_k;
    sparams.min_p = min_p;
    sparams.repetition_penalty = rep_penalty;
    sparams.max_tokens = max_tokens > 0 ? max_tokens : 512;

    auto ctx = std::make_unique<JavaCallbackContext>();
    if (callback) {
        ctx->global_callback = env->NewGlobalRef(callback);
    }

    auto token_fn = [](const mllm_token_chunk_t* chunk, void* user_data) -> bool {
        auto* c = static_cast<JavaCallbackContext*>(user_data);
        if (!c || !c->global_callback || !g_jvm) return true;

        JNIEnv* thread_env = nullptr;
        bool attached = false;
        jint res = g_jvm->GetEnv(reinterpret_cast<void**>(&thread_env), JNI_VERSION_1_6);
        if (res == JNI_EDETACHED) {
            if (g_jvm->AttachCurrentThread(&thread_env, nullptr) == JNI_OK) {
                attached = true;
            }
        }

        bool continue_gen = true;
        if (thread_env) {
            jclass cb_class = thread_env->GetObjectClass(c->global_callback);
            if (cb_class) {
                jmethodID mid = thread_env->GetMethodID(
                    cb_class,
                    "onToken",
                    "(Ljava/lang/String;IZF)Z"
                );

                if (mid) {
                    jstring jtext = thread_env->NewStringUTF((chunk && chunk->text) ? chunk->text : "");
                    jboolean ret = thread_env->CallBooleanMethod(
                        c->global_callback,
                        mid,
                        jtext,
                        chunk ? chunk->token_id : 0,
                        chunk ? chunk->is_special : false,
                        chunk ? chunk->logprob : 0.0f
                    );
                    if (thread_env->ExceptionCheck()) {
                        thread_env->ExceptionClear();
                        continue_gen = false;
                    } else {
                        continue_gen = (ret == JNI_TRUE);
                    }
                    if (jtext) {
                        thread_env->DeleteLocalRef(jtext);
                    }
                }
                thread_env->DeleteLocalRef(cb_class);
            }
        }
        return continue_gen;
    };

    auto complete_fn = [](mllm_status_t status, const mllm_metrics_t* metrics, void* user_data) {
        std::unique_ptr<JavaCallbackContext> c(static_cast<JavaCallbackContext*>(user_data));
        if (!c || !g_jvm) {
            return;
        }

        if (c->global_callback) {
            JNIEnv* thread_env = nullptr;
            bool attached = false;
            jint res = g_jvm->GetEnv(reinterpret_cast<void**>(&thread_env), JNI_VERSION_1_6);
            if (res == JNI_EDETACHED) {
                if (g_jvm->AttachCurrentThread(&thread_env, nullptr) == JNI_OK) {
                    attached = true;
                }
            }

            if (thread_env) {
                jclass cb_class = thread_env->GetObjectClass(c->global_callback);
                if (cb_class) {
                    jmethodID mid = thread_env->GetMethodID(
                        cb_class,
                        "onComplete",
                        "(ILjava/lang/String;IIDDD)V"
                    );

                    if (mid) {
                        jstring jmsg = thread_env->NewStringUTF(status.message ? status.message : "");
                        thread_env->CallVoidMethod(
                            c->global_callback,
                            mid,
                            status.code,
                            jmsg,
                            metrics ? metrics->prompt_tokens : 0,
                            metrics ? metrics->completion_tokens : 0,
                            metrics ? metrics->ttft_ms : 0.0,
                            metrics ? metrics->tps : 0.0,
                            metrics ? metrics->peak_rss_mb : 0.0
                        );
                        if (thread_env->ExceptionCheck()) {
                            thread_env->ExceptionClear();
                        }
                        if (jmsg) {
                            thread_env->DeleteLocalRef(jmsg);
                        }
                    }
                    thread_env->DeleteLocalRef(cb_class);
                }
                thread_env->DeleteGlobalRef(c->global_callback);
                c->global_callback = nullptr;
            }

            if (attached) {
                g_jvm->DetachCurrentThread();
            }
        }
    };

    mllm_gen_handle_t handle = 0;
    mllm_status_t st = mllm_generate(
        static_cast<mllm_engine_t>(engine_handle),
        static_cast<mllm_session_t>(session_handle),
        prompt_guard.c_str(),
        &sparams,
        token_fn,
        complete_fn,
        ctx.get(),
        &handle
    );

    if (st.code != 0) {
        // ctx is safely destroyed by std::unique_ptr, freeing global_callback and memory
        throw_java_exception(env, st.code, st.message);
        return 0;
    }

    // Hand over ownership to complete_fn
    ctx.release();
    return static_cast<jlong>(handle);
}

JNIEXPORT void JNICALL
Java_ai_codesec_mobilellm_NativeBridge_nativeCancel(
    JNIEnv* env,
    jclass /*clazz*/,
    jlong engine_handle,
    jlong gen_handle
) {
    mllm_status_t st = mllm_cancel(
        static_cast<mllm_engine_t>(engine_handle),
        static_cast<mllm_gen_handle_t>(gen_handle)
    );
    if (st.code != 0) {
        throw_java_exception(env, st.code, st.message);
    }
}

JNIEXPORT jstring JNICALL
Java_ai_codesec_mobilellm_NativeBridge_nativeInspectModel(
    JNIEnv* env,
    jclass /*clazz*/,
    jstring model_path
) {
    if (!model_path) {
        throw_java_exception(env, 1, "model_path cannot be null");
        return nullptr;
    }
    JStringGuard path_guard(env, model_path);
    char buf[1024];
    mllm_status_t st = mllm_model_inspect(path_guard.c_str(), buf, sizeof(buf));
    if (st.code != 0) {
        throw_java_exception(env, st.code, st.message);
        return nullptr;
    }
    return env->NewStringUTF(buf);
}

JNIEXPORT jstring JNICALL
Java_ai_codesec_mobilellm_NativeBridge_nativeGetVersion(JNIEnv* env, jclass /*clazz*/) {
    char buf[64];
    mllm_get_version(buf, sizeof(buf));
    return env->NewStringUTF(buf);
}

JNIEXPORT jstring JNICALL
Java_ai_codesec_mobilellm_NativeBridge_nativeDetectHardware(JNIEnv* env, jclass /*clazz*/) {
    char buf[512];
    mllm_detect_hardware(buf, sizeof(buf));
    return env->NewStringUTF(buf);
}

} // extern "C"

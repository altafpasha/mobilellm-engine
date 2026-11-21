#include <jni.h>
#include "mobilellm/c_api.h"
#include <string>
#include <memory>
#include <cstring>
#include <vector>
#include <chrono>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <dlfcn.h>
#include <android/log.h>

#define LOG_CL(...) __android_log_print(ANDROID_LOG_INFO, "MobileLLM_OpenCL", __VA_ARGS__)

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

static jstring make_safe_utf8_jstring(JNIEnv* env, const char* str) {
    if (!str || !*str) {
        return env->NewStringUTF("");
    }
    const auto* bytes = reinterpret_cast<const unsigned char*>(str);
    std::string safe;
    safe.reserve(std::strlen(str));
    size_t i = 0;
    size_t len = std::strlen(str);

    while (i < len) {
        unsigned char c = bytes[i];
        if (c < 0x80) {
            if (c == 0) break;
            safe.push_back(static_cast<char>(c));
            i++;
        } else if ((c & 0xE0) == 0xC0) {
            if (i + 1 < len && (bytes[i + 1] & 0xC0) == 0x80 && (c >= 0xC2)) {
                safe.push_back(static_cast<char>(bytes[i]));
                safe.push_back(static_cast<char>(bytes[i + 1]));
                i += 2;
            } else {
                safe.append("\xEF\xBF\xBD");
                i++;
            }
        } else if ((c & 0xF0) == 0xE0) {
            if (i + 2 < len &&
                (bytes[i + 1] & 0xC0) == 0x80 &&
                (bytes[i + 2] & 0xC0) == 0x80) {
                if ((c == 0xE0 && bytes[i + 1] < 0xA0) ||
                    (c == 0xED && bytes[i + 1] >= 0xA0)) {
                    safe.append("\xEF\xBF\xBD");
                    i++;
                } else {
                    safe.push_back(static_cast<char>(bytes[i]));
                    safe.push_back(static_cast<char>(bytes[i + 1]));
                    safe.push_back(static_cast<char>(bytes[i + 2]));
                    i += 3;
                }
            } else {
                safe.append("\xEF\xBF\xBD");
                i++;
            }
        } else if ((c & 0xF8) == 0xF0) {
            if (i + 3 < len &&
                (bytes[i + 1] & 0xC0) == 0x80 &&
                (bytes[i + 2] & 0xC0) == 0x80 &&
                (bytes[i + 3] & 0xC0) == 0x80) {
                if ((c == 0xF0 && bytes[i + 1] < 0x90) || (c > 0xF4) ||
                    (c == 0xF4 && bytes[i + 1] > 0x8F)) {
                    safe.append("\xEF\xBF\xBD");
                    i++;
                } else {
                    safe.push_back(static_cast<char>(bytes[i]));
                    safe.push_back(static_cast<char>(bytes[i + 1]));
                    safe.push_back(static_cast<char>(bytes[i + 2]));
                    safe.push_back(static_cast<char>(bytes[i + 3]));
                    i += 4;
                }
            } else {
                safe.append("\xEF\xBF\xBD");
                i++;
            }
        } else {
            safe.append("\xEF\xBF\xBD");
            i++;
        }
    }

    return env->NewStringUTF(safe.c_str());
}

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

JNIEXPORT void JNICALL
Java_ai_codesec_mobilellm_NativeBridge_nativeSwitchBackend(
    JNIEnv* env,
    jclass /*clazz*/,
    jlong engine_handle,
    jint backend
) {
    if (engine_handle == 0) return;
    mllm_status_t st = mllm_engine_switch_backend(
        static_cast<mllm_engine_t>(engine_handle),
        backend
    );
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
    cfg.n_gpu_layers = -1;

    mllm_model_t model = 0;
    mllm_status_t st = mllm_model_load(static_cast<mllm_engine_t>(engine_handle), &cfg, &model);
    if (st.code != 0) {
        throw_java_exception(env, st.code, st.message);
        return 0;
    }
    return static_cast<jlong>(model);
}

JNIEXPORT jlong JNICALL
Java_ai_codesec_mobilellm_NativeBridge_nativeLoadModelWithLayers(
    JNIEnv* env,
    jclass /*clazz*/,
    jlong engine_handle,
    jstring model_path,
    jint backend,
    jint context_length,
    jint n_threads,
    jboolean use_mmap,
    jfloat memory_budget_mb,
    jint n_gpu_layers
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
    cfg.n_gpu_layers = n_gpu_layers;

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
    mllm_session_config_t cfg = {};
    cfg.max_context_tokens = max_context_tokens > 0 ? max_context_tokens : 2048;
    cfg.enable_prefix_caching = enable_prefix_caching;
    cfg.n_batch = 512;
    cfg.n_ubatch = 0;
    cfg.flash_attn = -1;

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

JNIEXPORT void JNICALL
Java_ai_codesec_mobilellm_NativeBridge_nativeResetSession(
    JNIEnv* env,
    jclass /*clazz*/,
    jlong engine_handle,
    jlong session_handle
) {
    mllm_status_t st = mllm_session_reset(
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
                    jstring jtext = make_safe_utf8_jstring(thread_env, (chunk && chunk->text) ? chunk->text : "");
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
                    jmethodID mid_detailed = thread_env->GetMethodID(
                        cb_class,
                        "onCompleteDetailed",
                        "(ILjava/lang/String;IIDDDDDDD)V"
                    );
                    if (mid_detailed) {
                        jstring jmsg = make_safe_utf8_jstring(thread_env, status.message ? status.message : "");
                        thread_env->CallVoidMethod(
                            c->global_callback,
                            mid_detailed,
                            status.code,
                            jmsg,
                            metrics ? metrics->prompt_tokens : 0,
                            metrics ? metrics->completion_tokens : 0,
                            metrics ? metrics->ttft_ms : 0.0,
                            metrics ? metrics->tps : 0.0,
                            metrics ? metrics->peak_rss_mb : 0.0,
                            metrics ? metrics->prompt_tps : 0.0,
                            metrics ? metrics->prefill_ms : 0.0,
                            metrics ? metrics->decode_ms : 0.0,
                            metrics ? metrics->total_ms : 0.0
                        );
                        if (thread_env->ExceptionCheck()) {
                            thread_env->ExceptionClear();
                        }
                        if (jmsg) {
                            thread_env->DeleteLocalRef(jmsg);
                        }
                    } else {
                        thread_env->ExceptionClear();
                        jmethodID mid = thread_env->GetMethodID(
                            cb_class,
                            "onComplete",
                            "(ILjava/lang/String;IIDDD)V"
                        );
                        if (mid) {
                            jstring jmsg = make_safe_utf8_jstring(thread_env, status.message ? status.message : "");
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

JNIEXPORT jstring JNICALL
Java_ai_codesec_mobilellm_NativeBridge_nativeGetGenerationDiagnostics(
    JNIEnv* env,
    jclass /*clazz*/,
    jlong engine_handle,
    jlong session_handle
) {
    char buf[4096];
    mllm_status_t st = mllm_get_generation_diagnostics(
        static_cast<mllm_engine_t>(engine_handle),
        static_cast<mllm_session_t>(session_handle),
        buf,
        sizeof(buf)
    );
    if (st.code != 0) {
        throw_java_exception(env, st.code, st.message);
        return nullptr;
    }
    return env->NewStringUTF(buf);
}

JNIEXPORT jlongArray JNICALL
Java_ai_codesec_mobilellm_NativeBridge_nativeGetProcessMemory(
    JNIEnv* env,
    jclass /*clazz*/,
    jlong engine_handle
) {
    mllm_process_memory_t mem;
    std::memset(&mem, 0, sizeof(mem));
    mllm_status_t st = mllm_get_process_memory(
        static_cast<mllm_engine_t>(engine_handle),
        &mem
    );
    if (st.code != 0) {
        return nullptr;
    }

    jlongArray result = env->NewLongArray(5);
    if (!result) return nullptr;

    jlong vals[5];
    vals[0] = static_cast<jlong>(mem.current_rss_bytes);
    vals[1] = static_cast<jlong>(mem.peak_rss_bytes);
    vals[2] = static_cast<jlong>(mem.vmsize_bytes);
    vals[3] = static_cast<jlong>(mem.model_allocated_bytes);
    vals[4] = static_cast<jlong>(mem.kv_allocated_bytes);

    env->SetLongArrayRegion(result, 0, 5, vals);
    return result;
}

JNIEXPORT jintArray JNICALL
Java_ai_codesec_mobilellm_NativeBridge_nativeGetSessionContextTelemetry(
    JNIEnv* env,
    jclass /*clazz*/,
    jlong engine_handle,
    jlong session_handle
) {
    mllm_context_telemetry_t telem;
    std::memset(&telem, 0, sizeof(telem));
    mllm_status_t st = mllm_get_session_telemetry(
        static_cast<mllm_engine_t>(engine_handle),
        static_cast<mllm_session_t>(session_handle),
        &telem
    );
    if (st.code != 0) {
        return nullptr;
    }

    jintArray result = env->NewIntArray(4);
    if (!result) return nullptr;

    jint vals[4];
    vals[0] = static_cast<jint>(telem.context_capacity);
    vals[1] = static_cast<jint>(telem.used_tokens);
    vals[2] = static_cast<jint>(telem.remaining_capacity);
    vals[3] = telem.is_available ? 1 : 0;

    env->SetIntArrayRegion(result, 0, 4, vals);
    return result;
}

JNIEXPORT jboolean JNICALL
Java_ai_codesec_mobilellm_NativeBridge_nativeSetSessionThreads(
    JNIEnv* /*env*/,
    jclass /*clazz*/,
    jlong engine_handle,
    jlong session_handle,
    jint n_threads
) {
    mllm_status_t st = mllm_set_session_threads(
        static_cast<mllm_engine_t>(engine_handle),
        static_cast<mllm_session_t>(session_handle),
        static_cast<uint32_t>(n_threads)
    );
    return (st.code == 0) ? JNI_TRUE : JNI_FALSE;
}

namespace {

typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef size_t cl_size_t;
typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_mem;
typedef void* cl_program;
typedef void* cl_kernel;
typedef void* cl_event;

#define MLLM_CL_SUCCESS 0
#define MLLM_CL_DEVICE_TYPE_GPU (1 << 2)
#define MLLM_CL_MEM_READ_ONLY (1 << 2)
#define MLLM_CL_MEM_WRITE_ONLY (1 << 1)
#define MLLM_CL_MEM_COPY_HOST_PTR (1 << 5)

#define MLLM_CL_PLATFORM_NAME 0x0902
#define MLLM_CL_PLATFORM_VENDOR 0x0903
#define MLLM_CL_PLATFORM_VERSION 0x0901

#define MLLM_CL_DEVICE_NAME 0x102B
#define MLLM_CL_DEVICE_VENDOR 0x102C
#define MLLM_CL_DRIVER_VERSION 0x102D
#define MLLM_CL_DEVICE_VERSION 0x102F
#define MLLM_CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define MLLM_CL_DEVICE_MAX_CLOCK_FREQUENCY 0x100C
#define MLLM_CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1004
#define MLLM_CL_DEVICE_EXTENSIONS 0x1030

typedef cl_int (*pfn_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*pfn_clGetPlatformInfo)(cl_platform_id, cl_uint, size_t, void*, size_t*);
typedef cl_int (*pfn_clGetDeviceIDs)(cl_platform_id, cl_ulong, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (*pfn_clGetDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*);
typedef cl_context (*pfn_clCreateContext)(const void*, cl_uint, const cl_device_id*, void*, void*, cl_int*);
typedef cl_command_queue (*pfn_clCreateCommandQueue)(cl_context, cl_device_id, cl_ulong, cl_int*);
typedef cl_mem (*pfn_clCreateBuffer)(cl_context, cl_ulong, size_t, void*, cl_int*);
typedef cl_program (*pfn_clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (*pfn_clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
typedef cl_kernel (*pfn_clCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_int (*pfn_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int (*pfn_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*pfn_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*pfn_clFinish)(cl_command_queue);
typedef cl_int (*pfn_clReleaseMemObject)(cl_mem);
typedef cl_int (*pfn_clReleaseKernel)(cl_kernel);
typedef cl_int (*pfn_clReleaseProgram)(cl_program);
typedef cl_int (*pfn_clReleaseCommandQueue)(cl_command_queue);
typedef cl_int (*pfn_clReleaseContext)(cl_context);

static std::string run_opencl_diagnostic_internal() {
    LOG_CL("Attempting to load OpenCL library...");
    void* handle = dlopen("libOpenCL.so", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) handle = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) handle = dlopen("/system/vendor/lib64/libOpenCL.so", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        const char* err_str = dlerror();
        std::string err_desc = err_str ? err_str : "library could not be loaded";
        LOG_CL("Failed to load libOpenCL.so: %s", err_desc.c_str());
        return "{\"success\":false,\"error\":\"Failed to load libOpenCL.so: " + err_desc + "\"}";
    }
    LOG_CL("OpenCL library loaded successfully (handle=%p). Resolving symbols...", handle);

#define LOAD_CL_SYM(name) \
    pfn_##name fn_##name = reinterpret_cast<pfn_##name>(dlsym(handle, #name)); \
    if (!fn_##name) { \
        LOG_CL("Failed to resolve OpenCL symbol: " #name); \
        dlclose(handle); \
        return "{\"success\":false,\"error\":\"Failed to resolve OpenCL symbol: " #name "\"}"; \
    }

    LOAD_CL_SYM(clGetPlatformIDs);
    LOAD_CL_SYM(clGetPlatformInfo);
    LOAD_CL_SYM(clGetDeviceIDs);
    LOAD_CL_SYM(clGetDeviceInfo);
    LOAD_CL_SYM(clCreateContext);
    LOAD_CL_SYM(clCreateCommandQueue);
    LOAD_CL_SYM(clCreateBuffer);
    LOAD_CL_SYM(clCreateProgramWithSource);
    LOAD_CL_SYM(clBuildProgram);
    LOAD_CL_SYM(clCreateKernel);
    LOAD_CL_SYM(clSetKernelArg);
    LOAD_CL_SYM(clEnqueueNDRangeKernel);
    LOAD_CL_SYM(clEnqueueReadBuffer);
    LOAD_CL_SYM(clFinish);
    LOAD_CL_SYM(clReleaseMemObject);
    LOAD_CL_SYM(clReleaseKernel);
    LOAD_CL_SYM(clReleaseProgram);
    LOAD_CL_SYM(clReleaseCommandQueue);
    LOAD_CL_SYM(clReleaseContext);
#undef LOAD_CL_SYM

    cl_uint num_platforms = 0;
    cl_platform_id platform = nullptr;
    cl_int err = fn_clGetPlatformIDs(1, &platform, &num_platforms);
    if (err != MLLM_CL_SUCCESS || !platform) {
        dlclose(handle);
        return "{\"success\":false,\"error\":\"No OpenCL platforms found (err=" + std::to_string(err) + ")\"}";
    }

    char plat_name[256] = {0};
    char plat_vendor[256] = {0};
    char plat_version[256] = {0};
    size_t info_ret_len = 0;
    fn_clGetPlatformInfo(platform, MLLM_CL_PLATFORM_NAME, sizeof(plat_name) - 1, plat_name, &info_ret_len);
    fn_clGetPlatformInfo(platform, MLLM_CL_PLATFORM_VENDOR, sizeof(plat_vendor) - 1, plat_vendor, &info_ret_len);
    fn_clGetPlatformInfo(platform, MLLM_CL_PLATFORM_VERSION, sizeof(plat_version) - 1, plat_version, &info_ret_len);

    cl_uint num_devices = 0;
    cl_device_id device = nullptr;
    err = fn_clGetDeviceIDs(platform, MLLM_CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);
    if (err != MLLM_CL_SUCCESS || !device) {
        dlclose(handle);
        return "{\"success\":false,\"error\":\"No OpenCL GPU devices found (err=" + std::to_string(err) + ")\"}";
    }

    char dev_name[256] = {0};
    char drv_version[256] = {0};
    char dev_version[256] = {0};
    char extensions[8192] = {0};
    cl_uint compute_units = 0;
    cl_uint max_clock = 0;
    size_t max_workgroup = 0;

    fn_clGetDeviceInfo(device, MLLM_CL_DEVICE_NAME, sizeof(dev_name) - 1, dev_name, &info_ret_len);
    fn_clGetDeviceInfo(device, MLLM_CL_DRIVER_VERSION, sizeof(drv_version) - 1, drv_version, &info_ret_len);
    fn_clGetDeviceInfo(device, MLLM_CL_DEVICE_VERSION, sizeof(dev_version) - 1, dev_version, &info_ret_len);
    fn_clGetDeviceInfo(device, MLLM_CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, &info_ret_len);
    fn_clGetDeviceInfo(device, MLLM_CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(max_clock), &max_clock, &info_ret_len);
    fn_clGetDeviceInfo(device, MLLM_CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_workgroup), &max_workgroup, &info_ret_len);
    fn_clGetDeviceInfo(device, MLLM_CL_DEVICE_EXTENSIONS, sizeof(extensions) - 1, extensions, &info_ret_len);

    std::string ext_str(extensions);
    bool fp16_supported = (ext_str.find("cl_khr_fp16") != std::string::npos);
    bool dot_product_supported = (ext_str.find("cl_qcom_dot_product8") != std::string::npos);
    bool subgroup_shuffle_supported = (ext_str.find("cl_qcom_subgroup_shuffle") != std::string::npos);
    LOG_CL("Platform: %s | Device: %s | Driver: %s", plat_name, dev_name, drv_version);
    LOG_CL("Features: fp16=%d, dot8=%d, shuffle=%d", fp16_supported ? 1 : 0, dot_product_supported ? 1 : 0, subgroup_shuffle_supported ? 1 : 0);

    cl_context ctx = fn_clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != MLLM_CL_SUCCESS || !ctx) {
        LOG_CL("clCreateContext failed (err=%d)", err);
        dlclose(handle);
        return "{\"success\":false,\"error\":\"clCreateContext failed (err=" + std::to_string(err) + ")\"}";
    }

    cl_command_queue queue = fn_clCreateCommandQueue(ctx, device, 0, &err);
    if (err != MLLM_CL_SUCCESS || !queue) {
        LOG_CL("clCreateCommandQueue failed (err=%d)", err);
        fn_clReleaseContext(ctx);
        dlclose(handle);
        return "{\"success\":false,\"error\":\"clCreateCommandQueue failed (err=" + std::to_string(err) + ")\"}";
    }

    const uint32_t K = 2048;
    const uint32_t N = 2048;

    const char* kernel_src =
        "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
        "__kernel void gemv_f16_f32_vec8(\n"
        "    __global const half8* weights, // [N, K/8]\n"
        "    __global const float* vec_in,  // [K]\n"
        "    __global float* vec_out,       // [N]\n"
        "    const uint K_over_8\n"
        ") {\n"
        "    uint row = get_global_id(0);\n"
        "    float sum0 = 0.0f, sum1 = 0.0f;\n"
        "    __global const half8* w_row = weights + row * K_over_8;\n"
        "    for (uint i = 0; i < K_over_8; i += 2) {\n"
        "        half8 w0 = w_row[i];\n"
        "        uint k0 = i * 8;\n"
        "        sum0 += (float)w0.s0 * vec_in[k0]   + (float)w0.s1 * vec_in[k0+1]\n"
        "              + (float)w0.s2 * vec_in[k0+2] + (float)w0.s3 * vec_in[k0+3]\n"
        "              + (float)w0.s4 * vec_in[k0+4] + (float)w0.s5 * vec_in[k0+5]\n"
        "              + (float)w0.s6 * vec_in[k0+6] + (float)w0.s7 * vec_in[k0+7];\n"
        "        half8 w1 = w_row[i + 1];\n"
        "        uint k1 = (i + 1) * 8;\n"
        "        sum1 += (float)w1.s0 * vec_in[k1]   + (float)w1.s1 * vec_in[k1+1]\n"
        "              + (float)w1.s2 * vec_in[k1+2] + (float)w1.s3 * vec_in[k1+3]\n"
        "              + (float)w1.s4 * vec_in[k1+4] + (float)w1.s5 * vec_in[k1+5]\n"
        "              + (float)w1.s6 * vec_in[k1+6] + (float)w1.s7 * vec_in[k1+7];\n"
        "    }\n"
        "    vec_out[row] = sum0 + sum1;\n"
        "}\n";

    LOG_CL("Building OpenCL kernel...");
    cl_program prg = fn_clCreateProgramWithSource(ctx, 1, &kernel_src, nullptr, &err);
    err = fn_clBuildProgram(prg, 1, &device, "-cl-fast-relaxed-math -cl-mad-enable", nullptr, nullptr);
    if (err != MLLM_CL_SUCCESS) {
        LOG_CL("clBuildProgram failed (err=%d)", err);
        fn_clReleaseProgram(prg);
        fn_clReleaseCommandQueue(queue);
        fn_clReleaseContext(ctx);
        dlclose(handle);
        return "{\"success\":false,\"error\":\"clBuildProgram failed (err=" + std::to_string(err) + ")\"}";
    }

    LOG_CL("clBuildProgram succeeded. Dispatching GEMV benchmark...");

    cl_kernel krn = fn_clCreateKernel(prg, "gemv_f16_f32_vec8", &err);

    std::vector<uint16_t> h_weights(N * K, 0x3c00); // 1.0 in FP16
    std::vector<float> h_in(K, 0.5f);
    std::vector<float> h_out(N, 0.0f);

    cl_mem d_weights = fn_clCreateBuffer(ctx, MLLM_CL_MEM_READ_ONLY | MLLM_CL_MEM_COPY_HOST_PTR, N * K * sizeof(uint16_t), h_weights.data(), &err);
    cl_mem d_in = fn_clCreateBuffer(ctx, MLLM_CL_MEM_READ_ONLY | MLLM_CL_MEM_COPY_HOST_PTR, K * sizeof(float), h_in.data(), &err);
    cl_mem d_out = fn_clCreateBuffer(ctx, MLLM_CL_MEM_WRITE_ONLY, N * sizeof(float), nullptr, &err);

    uint32_t K_over_8 = K / 8;
    fn_clSetKernelArg(krn, 0, sizeof(cl_mem), &d_weights);
    fn_clSetKernelArg(krn, 1, sizeof(cl_mem), &d_in);
    fn_clSetKernelArg(krn, 2, sizeof(cl_mem), &d_out);
    fn_clSetKernelArg(krn, 3, sizeof(uint32_t), &K_over_8);

    size_t global_size = N;
    size_t local_size = 64; // Adreno 64-wide wave

    // Warmup
    for (int w = 0; w < 10; ++w) {
        fn_clEnqueueNDRangeKernel(queue, krn, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);
    }
    fn_clFinish(queue);

    // Benchmark 100 iterations
    const int ITERS = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; ++it) {
        fn_clEnqueueNDRangeKernel(queue, krn, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);
    }
    fn_clFinish(queue);
    auto end = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double per_layer_ms = (total_us / ITERS) / 1000.0;

    fn_clEnqueueReadBuffer(queue, d_out, 1, 0, N * sizeof(float), h_out.data(), 0, nullptr, nullptr);

    bool math_pass = (std::abs(h_out[0] - 1024.0f) < 1.0f);

    double bytes_transferred = static_cast<double>(N) * K * sizeof(uint16_t) + K * sizeof(float) + N * sizeof(float);
    double effective_bandwidth_gb_s = (bytes_transferred / (per_layer_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

    double projected_28l_decode_ms = per_layer_ms * 28 * 4;
    double projected_decode_tok_s = (1000.0 / projected_28l_decode_ms);

    fn_clReleaseMemObject(d_weights);
    fn_clReleaseMemObject(d_in);
    fn_clReleaseMemObject(d_out);
    fn_clReleaseKernel(krn);
    fn_clReleaseProgram(prg);
    fn_clReleaseCommandQueue(queue);
    fn_clReleaseContext(ctx);
    dlclose(handle);

    LOG_CL("OpenCL benchmark completed: per_layer_ms=%.3f ms, bandwidth=%.2f GB/s, math_pass=%d",
           per_layer_ms, effective_bandwidth_gb_s, math_pass ? 1 : 0);

    std::ostringstream json;
    json << std::fixed << std::setprecision(3);
    json << "{"
         << "\"success\":true,"
         << "\"platform\":\"" << plat_name << "\","
         << "\"device\":\"" << dev_name << "\","
         << "\"opencl_version\":\"" << dev_version << "\","
         << "\"driver_version\":\"" << drv_version << "\","
         << "\"compute_units\":" << compute_units << ","
         << "\"max_clock_mhz\":" << max_clock << ","
         << "\"fp16_supported\":" << (fp16_supported ? "true" : "false") << ","
         << "\"dot_product_supported\":" << (dot_product_supported ? "true" : "false") << ","
         << "\"subgroup_shuffle_supported\":" << (subgroup_shuffle_supported ? "true" : "false") << ","
         << "\"math_verification\":\"" << (math_pass ? "PASS (Bit-accurate: 1024.0)" : "FAIL") << "\","
         << "\"per_gemv_ms\":" << per_layer_ms << ","
         << "\"bandwidth_gbps\":" << effective_bandwidth_gb_s << ","
         << "\"projected_decode_ms\":" << projected_28l_decode_ms << ","
         << "\"projected_tok_s\":" << projected_decode_tok_s << ","
         << "\"raw_report\":\"Device: " << dev_name << " | Driver: " << drv_version
         << " | Latency: " << per_layer_ms << " ms | Bandwidth: " << effective_bandwidth_gb_s
         << " GB/s | Bit-accuracy: " << (math_pass ? "PASS" : "FAIL") << "\""
         << "}";

    return json.str();
}

} // namespace

JNIEXPORT jstring JNICALL
Java_ai_codesec_mobilellm_NativeBridge_nativeRunOpenCLDiagnostic(
    JNIEnv* env,
    jclass /*clazz*/
) {
    std::string report = run_opencl_diagnostic_internal();
    return env->NewStringUTF(report.c_str());
}

} // extern "C"

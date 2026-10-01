/*
 * ggml_vit.c - Vision Transformer inference using GGML operations.
 *
 * Implements the full DINOv2/v3 ViT forward pass:
 *   1. Patch embedding via conv2d
 *   2. Token assembly (CLS + patches + registers)
 *   3. Transformer blocks with RoPE attention and MLP
 *   4. Final LayerNorm and output flattening
 */

#include "ggml_vit.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <time.h>
#endif

/* ======================================================================== */
/* Internal structures                                                       */
/* ======================================================================== */

/* Per-layer weight tensors.
 * The LayerNorm affine params (norm1/norm2 w,b) are folded into qkv/fc1
 * weights+biases at load time, and LayerScale gammas into proj/fc2:
 * removes 6 elementwise passes per layer. */
typedef struct {
    struct ggml_tensor * qkv_w;        /* [H, 3*H], norm1 affine folded in */
    struct ggml_tensor * qkv_b;        /* [3*H] */
    struct ggml_tensor * proj_w;       /* [H, H], ls1 folded in */
    struct ggml_tensor * proj_b;       /* [H] */
    struct ggml_tensor * fc1_w;        /* [H, I], norm2 affine folded in */
    struct ggml_tensor * fc1_b;        /* [I] */
    struct ggml_tensor * fc2_w;        /* [I, H], ls2 folded in */
    struct ggml_tensor * fc2_b;        /* [H] */
} vit_layer_weights_t;

struct ggml_vit_model {
    vit_config_t config;
    int head_dim;
    int grid_h;
    int grid_w;
    int seq_len;
    int output_dim;

    /* Backend */
    ggml_backend_t backend;

    /* CUDA backend: keep activations in F16 between GEMMs. Halves the
     * elementwise bandwidth of the whole network and lets cuBLAS write GEMM
     * outputs directly as F16 (skipping both hidden conversion passes).
     * CPU builds stay all-F32 (CPU kernels are F32-only). */
    int use_f16;

    /* Weight context and buffer */
    struct ggml_context * weight_ctx;
    struct ggml_backend_buffer * weight_buf;

    /* Global weights */
    struct ggml_tensor * patch_embed_w;  /* [PS, PS, 3, H] */
    struct ggml_tensor * patch_embed_b;  /* [H] */
    /* CLS + register tokens, pre-broadcast over the batch dim: [H, 1+n_reg, B].
     * Constant for every image, so a plain weight beats a per-forward repeat. */
    struct ggml_tensor * token_prefix_1; /* [H, 1+n_reg, 1] */
    struct ggml_tensor * token_prefix_B; /* [H, 1+n_reg, max_batch] */
    struct ggml_tensor * norm_w;         /* [H] */
    struct ggml_tensor * norm_b;         /* [H] */

    /* Per-layer weights */
    vit_layer_weights_t * layers;

    /* DINOv3 RoPE via GGML_OP_ROPE vision mode: quantized 2D positions
     * [4*seq_len] I32 (y then x per token; CLS/register rows are 0 = identity)
     * and freq factors [head_dim/2] F32 undoing the position quantization
     * scale, so the kernel computes theta = 2*pi*coord*base^(-4p/D). */
    struct ggml_tensor * rope_pos;
    struct ggml_tensor * rope_ff;

    /* Graph allocator */
    ggml_gallocr_t galloc;

    /* Prebuilt compute graph (input size is fixed): built once, reused for
     * every inference. Stable topology enables CUDA graph capture. */
    struct ggml_context * compute_ctx;
    struct ggml_cgraph * graph;
    struct ggml_tensor * input_tensor;
    struct ggml_tensor * output_tensor;

    /* Prebuilt batched graph (max_batch images per forward pass) */
    int max_batch;
    ggml_gallocr_t galloc_batch;
    struct ggml_context * compute_ctx_batch;
    struct ggml_cgraph * graph_batch;
    struct ggml_tensor * input_batch;
    struct ggml_tensor * output_batch;
};

/* Helper: upload a host F32 buffer into a tensor, converting to the tensor's
 * dtype (F32 or F16). Works for both CPU and GPU (device) buffers. */
static int upload_tensor(struct ggml_tensor * t, const float * buf) {
    size_t n = (size_t)ggml_nelements(t);
    if (t->type == GGML_TYPE_F16) {
        ggml_fp16_t * h = (ggml_fp16_t *)malloc(n * sizeof(ggml_fp16_t));
        if (!h) return -1;
        ggml_fp32_to_fp16_row(buf, h, (int64_t)n);
        ggml_backend_tensor_set(t, h, 0, n * sizeof(ggml_fp16_t));
        free(h);
    } else {
        ggml_backend_tensor_set(t, buf, 0, n * sizeof(float));
    }
    return 0;
}

/* Helper: read n raw F32 values from file into a malloc'd buffer. */
static float * read_raw(FILE * f, size_t n) {
    float * buf = (float *)malloc(n * sizeof(float));
    if (!buf) return NULL;
    if (fread(buf, sizeof(float), n, f) != n) { free(buf); return NULL; }
    return buf;
}

/* Helper: read a full F32 tensor from file and upload it. */
static int read_tensor(FILE * f, struct ggml_tensor * t) {
    float * buf = read_raw(f, (size_t)ggml_nelements(t));
    if (!buf) return -1;
    int ret = upload_tensor(t, buf);
    free(buf);
    return ret;
}

/* Helper: number of logical CPU cores for the compute threadpool */
static int get_num_cores(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    int n = (int)si.dwNumberOfProcessors;
#else
    int n = (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
    return n > 0 ? n : 4;
}

/* Defined after build_forward below. */
static int build_compute_graph(ggml_vit_model_t * model);

/* ======================================================================== */
/* Model creation                                                            */
/* ======================================================================== */

ggml_vit_model_t* ggml_vit_create(const vit_config_t* config) {
    ggml_vit_model_t * model = (ggml_vit_model_t *)calloc(1, sizeof(ggml_vit_model_t));
    if (!model) return NULL;

    model->config = *config;
    model->head_dim = config->hidden_size / config->num_heads;
    model->grid_h = config->input_height / config->patch_size;
    model->grid_w = config->input_width / config->patch_size;
    model->seq_len = config->has_cls_token + model->grid_h * model->grid_w + config->num_register_tokens;
    model->output_dim = model->seq_len * config->hidden_size;
    /* Default to single-image graphs: per-image latency is lower than batched
     * on GPUs already saturated at B=1; override via config or GGML_VIT_MAX_BATCH. */
    model->max_batch = config->max_batch > 0 ? config->max_batch : 1;
    {
        const char * mb = getenv("GGML_VIT_MAX_BATCH");
        if (mb && atoi(mb) > 0) model->max_batch = atoi(mb);
    }

    /* Initialize backend: CUDA if compiled in and available, else CPU */
    ggml_cpu_init();
#ifdef GGML_USE_CUDA
    if (!getenv("GGML_VIT_FORCE_CPU")) {
        model->backend = ggml_backend_cuda_init(0);
        if (model->backend) {
            fprintf(stderr, "ggml_vit: using CUDA backend (device 0)\n");
            /* F16 activation stream between GEMMs (CUDA kernels patched for it) */
            model->use_f16 = 1;
        } else {
            fprintf(stderr, "ggml_vit: CUDA init failed, falling back to CPU\n");
        }
    }
#endif
    if (!model->backend) {
        model->backend = ggml_backend_cpu_init();
        if (!model->backend) { free(model); return NULL; }
        ggml_backend_cpu_set_n_threads(model->backend, get_num_cores());
    }

    /* Allocate per-layer weight structs */
    model->layers = (vit_layer_weights_t *)calloc(config->num_layers, sizeof(vit_layer_weights_t));
    if (!model->layers) { ggml_vit_destroy(model); return NULL; }

    /* Create weight context (no_alloc = true, backend allocates via ggml_backend_alloc_ctx_tensors) */
    struct ggml_init_params ctx_params = {
        .mem_size   = (size_t)1024 * 1024 * 1024, /* 1 GB should be enough for weights */
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    model->weight_ctx = ggml_init(ctx_params);
    if (!model->weight_ctx) { ggml_vit_destroy(model); return NULL; }

    int H = config->hidden_size;
    int PS = config->patch_size;
    int n_reg = config->num_register_tokens;
    int n_heads = config->num_heads;
    int head_dim = model->head_dim;
    int I = config->intermediate_size;
    int prefix_len = config->has_cls_token + n_reg;
    enum ggml_type act = model->use_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    struct ggml_context * ctx = model->weight_ctx;

    /* Create weight tensors. Large matmul/conv weights are stored as F16:
     * halves bandwidth and enables tensor-core GEMM on CUDA. Biases and token
     * constants follow the activation dtype so adds/concats stay same-type. */
    model->patch_embed_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, PS, PS, 3, H);
    model->patch_embed_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    ggml_set_name(model->patch_embed_w, "patch_embed_w");
    ggml_set_name(model->patch_embed_b, "patch_embed_b");

    model->token_prefix_1 = ggml_new_tensor_3d(ctx, act, H, prefix_len, 1);
    ggml_set_name(model->token_prefix_1, "token_prefix_1");
    if (model->max_batch > 1) {
        model->token_prefix_B = ggml_new_tensor_3d(ctx, act, H, prefix_len, model->max_batch);
        ggml_set_name(model->token_prefix_B, "token_prefix_B");
    }

    model->norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    model->norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    ggml_set_name(model->norm_w, "norm_w");
    ggml_set_name(model->norm_b, "norm_b");

    /* Per-layer weights (norm/LayerScale params are folded in at load time) */
    for (int l = 0; l < config->num_layers; l++) {
        vit_layer_weights_t * L = &model->layers[l];
        char name[64];

        /* QKV weight transposed for ggml_mul_mat: [H, 3*H] */
        L->qkv_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, H, 3 * H);
        L->qkv_b = ggml_new_tensor_1d(ctx, act, 3 * H);
        /* Proj weight transposed: [H, H] */
        L->proj_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, H, H);
        L->proj_b = ggml_new_tensor_1d(ctx, act, H);
        /* FC1 weight transposed: [H, I] */
        L->fc1_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, H, I);
        L->fc1_b = ggml_new_tensor_1d(ctx, act, I);
        /* FC2 weight transposed: [I, H] */
        L->fc2_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, I, H);
        L->fc2_b = ggml_new_tensor_1d(ctx, act, H);

        snprintf(name, sizeof(name), "layer_%d_qkv_w", l);  ggml_set_name(L->qkv_w, name);
        snprintf(name, sizeof(name), "layer_%d_proj_w", l); ggml_set_name(L->proj_w, name);
        snprintf(name, sizeof(name), "layer_%d_fc1_w", l);  ggml_set_name(L->fc1_w, name);
        snprintf(name, sizeof(name), "layer_%d_fc2_w", l);  ggml_set_name(L->fc2_w, name);
    }

    /* DINOv3 RoPE via the GGML vision rope mode: positions quantized to I32.
     * Patch center coords normalized to [-1, +1]; the angle
     * 2*pi*coord*base^(-4p/D) is expressed as pos*theta_scale^p / ff with
     * pos = round(2*pi*coord*POS_SCALE) and ff = POS_SCALE (quantization
     * error <= pi/POS_SCALE ~ 1.5e-6 rad, below F32 cos/sin precision).
     * Token order matches HF DINOv3: [CLS, registers, patches]; CLS/register
     * rows keep pos = 0 (rotation by angle 0 = identity). */
    model->rope_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t)4 * model->seq_len);
    model->rope_ff  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim / 2);
    ggml_set_name(model->rope_pos, "rope_pos");
    ggml_set_name(model->rope_ff, "rope_ff");

    /* Allocate weight tensors on the backend */
    model->weight_buf = ggml_backend_alloc_ctx_tensors(ctx, model->backend);
    if (!model->weight_buf) { ggml_vit_destroy(model); return NULL; }

    /* Fill the RoPE position/freq-factor tables (after allocation). */
    {
        const int64_t S = model->seq_len;
        const int64_t n_prefix = config->has_cls_token + n_reg;
        const float POS_SCALE = 2097152.0f; /* 2^21; max |2*pi*coord*POS_SCALE| ~ 13.2e6 fits I32 */
        int32_t * pos = (int32_t *)malloc((size_t)4 * S * sizeof(int32_t));
        float * ff = (float *)malloc((size_t)(head_dim / 2) * sizeof(float));
        if (!pos || !ff) {
            free(pos); free(ff);
            ggml_vit_destroy(model);
            return NULL;
        }
        memset(pos, 0, (size_t)4 * S * sizeof(int32_t));
        const float two_pi = 6.28318530717958647692f;
        for (int row = 0; row < model->grid_h; row++) {
            for (int col = 0; col < model->grid_w; col++) {
                int64_t t = n_prefix + (int64_t)row * model->grid_w + col;
                float cy = 2.0f * (row + 0.5f) / model->grid_h - 1.0f;
                float cx = 2.0f * (col + 0.5f) / model->grid_w - 1.0f;
                pos[t]             = (int32_t)lrintf(two_pi * cy * POS_SCALE);
                pos[S + t]         = (int32_t)lrintf(two_pi * cx * POS_SCALE);
            }
        }
        for (int p = 0; p < head_dim / 2; p++) ff[p] = POS_SCALE;
        ggml_backend_tensor_set(model->rope_pos, pos, 0, (size_t)4 * S * sizeof(int32_t));
        ggml_backend_tensor_set(model->rope_ff, ff, 0, (size_t)(head_dim / 2) * sizeof(float));
        free(pos);
        free(ff);
    }

    /* Initialize graph allocator on the backend's buffer type */
    model->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model->backend));

    /* Build the compute graph once: input size is fixed, so the graph and
     * its allocation can be reused for every inference. */
    if (build_compute_graph(model) != 0) {
        ggml_vit_destroy(model);
        return NULL;
    }

    return model;
}

/* ======================================================================== */
/* Weight loading                                                            */
/* ======================================================================== */

int ggml_vit_load_weights(ggml_vit_model_t* model, const char* path) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ggml_vit: cannot open %s\n", path); return -1; }

    int n_layers = model->config.num_layers;
    const size_t H = (size_t)model->config.hidden_size;
    const size_t I = (size_t)model->config.intermediate_size;

    /* All 2D/4D weights are read as-is: PyTorch row-major [out, in] (or [OC,IC,KH,KW])
     * memory layout matches the GGML ne0-contiguous layout of the created tensors. */
    if (read_tensor(f, model->patch_embed_w) != 0) goto fail;
    if (read_tensor(f, model->patch_embed_b) != 0) goto fail;

    /* CLS + register tokens: read the raw values and fill the pre-broadcast
     * prefix tensors ([H, 1+n_reg, B], dtype = activation type). */
    {
        const int n_reg = model->config.num_register_tokens;
        const int prefix_len = model->config.has_cls_token + n_reg;
        const size_t one = (size_t)prefix_len * H;
        float * tok = read_raw(f, one);
        if (!tok) goto fail;
        int ok = upload_tensor(model->token_prefix_1, tok) == 0;
        if (ok && model->token_prefix_B) {
            /* batch prefix = the same tokens repeated along the batch dim */
            float * tokB = (float *)malloc(one * (size_t)model->max_batch * sizeof(float));
            if (!tokB) { free(tok); goto fail; }
            for (int b = 0; b < model->max_batch; b++) {
                memcpy(tokB + (size_t)b * one, tok, one * sizeof(float));
            }
            ok = upload_tensor(model->token_prefix_B, tokB) == 0;
            free(tokB);
        }
        free(tok);
        if (!ok) goto fail;
    }

    /* Per-layer weights. LayerNorm affine params are folded into the following
     * GEMM (W'(x) = W(w_n*x + b_n) => scale W columns by w_n, add W*b_n to the
     * bias) and LayerScale gammas into the preceding GEMM's output rows:
     * saves 6 full elementwise passes over the activations per layer. */
    for (int l = 0; l < n_layers; l++) {
        vit_layer_weights_t * L = &model->layers[l];

        float * norm1_w = read_raw(f, H);
        float * norm1_b = read_raw(f, H);
        float * qkv_w   = read_raw(f, 3 * H * H);
        float * qkv_b   = read_raw(f, 3 * H);
        float * proj_w  = read_raw(f, H * H);
        float * proj_b  = read_raw(f, H);
        float * norm2_w = read_raw(f, H);
        float * norm2_b = read_raw(f, H);
        float * fc1_w   = read_raw(f, I * H);
        float * fc1_b   = read_raw(f, I);
        float * fc2_w   = read_raw(f, H * I);
        float * fc2_b   = read_raw(f, H);
        float * ls1     = read_raw(f, H);
        float * ls2     = read_raw(f, H);

        int ok = norm1_w && norm1_b && qkv_w && qkv_b && proj_w && proj_b &&
                 norm2_w && norm2_b && fc1_w && fc1_b && fc2_w && fc2_b && ls1 && ls2;

        if (ok) {
            /* Fold norm1 affine into qkv: b += W*b_n first (uses original W),
             * then scale W columns by w_n. Weights are [out][in] row-major. */
            for (size_t o = 0; o < 3 * H; o++) {
                float acc = 0.0f;
                float * row = qkv_w + o * H;
                for (size_t i = 0; i < H; i++) acc += row[i] * norm1_b[i];
                qkv_b[o] += acc;
                for (size_t i = 0; i < H; i++) row[i] *= norm1_w[i];
            }
            /* Fold ls1 into proj output rows */
            for (size_t o = 0; o < H; o++) {
                float g = ls1[o];
                float * row = proj_w + o * H;
                for (size_t i = 0; i < H; i++) row[i] *= g;
                proj_b[o] *= g;
            }
            /* Fold norm2 affine into fc1 */
            for (size_t o = 0; o < I; o++) {
                float acc = 0.0f;
                float * row = fc1_w + o * H;
                for (size_t i = 0; i < H; i++) acc += row[i] * norm2_b[i];
                fc1_b[o] += acc;
                for (size_t i = 0; i < H; i++) row[i] *= norm2_w[i];
            }
            /* Fold ls2 into fc2 output rows */
            for (size_t o = 0; o < H; o++) {
                float g = ls2[o];
                float * row = fc2_w + o * I;
                for (size_t i = 0; i < I; i++) row[i] *= g;
                fc2_b[o] *= g;
            }

            ok = upload_tensor(L->qkv_w, qkv_w) == 0 &&
                 upload_tensor(L->qkv_b, qkv_b) == 0 &&
                 upload_tensor(L->proj_w, proj_w) == 0 &&
                 upload_tensor(L->proj_b, proj_b) == 0 &&
                 upload_tensor(L->fc1_w, fc1_w) == 0 &&
                 upload_tensor(L->fc1_b, fc1_b) == 0 &&
                 upload_tensor(L->fc2_w, fc2_w) == 0 &&
                 upload_tensor(L->fc2_b, fc2_b) == 0;
        }

        free(norm1_w); free(norm1_b); free(qkv_w); free(qkv_b);
        free(proj_w); free(proj_b); free(norm2_w); free(norm2_b);
        free(fc1_w); free(fc1_b); free(fc2_w); free(fc2_b);
        free(ls1); free(ls2);

        if (!ok) goto fail;
    }

    /* Final norm */
    if (read_tensor(f, model->norm_w) != 0) goto fail;
    if (read_tensor(f, model->norm_b) != 0) goto fail;

    fclose(f);
    return 0;

fail:
    fclose(f);
    fprintf(stderr, "ggml_vit: failed to read weights from %s\n", path);
    return -1;
}

/* ======================================================================== */
/* Graph building                                                            */
/* ======================================================================== */

/* Apply LayerNorm: norm(x, eps) * w + b (F32 in/out; used for the final norm) */
static struct ggml_tensor * apply_layer_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * w,
    struct ggml_tensor * b,
    float eps)
{
    struct ggml_tensor * normed = ggml_norm(ctx, x, eps);
    /* w and b are [H], normed is [H, seq_len] -> broadcasting works */
    struct ggml_tensor * scaled = ggml_mul(ctx, normed, w);
    return ggml_add(ctx, scaled, b);
}

/* GEMM with the result kept in the activation dtype (F16 on CUDA: cuBLAS
 * writes F16 directly, skipping the hidden F32 conversion round trip). */
static struct ggml_tensor * mm_act(
    ggml_vit_model_t * model,
    struct ggml_context * ctx,
    struct ggml_tensor * w,
    struct ggml_tensor * x)
{
    if (model->use_f16) {
        return ggml_mul_mat_out(ctx, w, x, GGML_TYPE_F16);
    }
    return ggml_mul_mat(ctx, w, x);
}

/* DINOv3 RoPE via the GGML vision rope mode.
 * x: [head_dim, n_heads, seq_len, B] view of the QKV output; the kernel
 * rotates half-pairs (p, p + D/2) with theta = pos(section) * base^(-4p/D),
 * which matches DINOv3's [y(D/4), x(D/4)] axial layout exactly. CLS/register
 * rows have pos 0 and pass through unchanged.
 * out_type lets the Q rope produce F32 directly from the F16 stream (flash
 * attention requires F32 Q) with no separate cast pass. */
static struct ggml_tensor * apply_rope_vision(
    ggml_vit_model_t * model,
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    enum ggml_type out_type)
{
    int sections[GGML_MROPE_SECTIONS] = { model->head_dim / 4, model->head_dim / 4, 0, 0 };
    return ggml_rope_multi_out(ctx, x, model->rope_pos, model->rope_ff,
                               model->head_dim / 2, sections,
                               GGML_ROPE_TYPE_VISION, 0,
                               model->config.rope_freq_base,
                               1.0f, 0.0f, 1.0f, 0.0f, 0.0f, out_type);
}

static struct ggml_tensor * build_forward(
    ggml_vit_model_t * model,
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    int batch)
{
    vit_config_t * cfg = &model->config;
    int H = cfg->hidden_size;
    int PS = cfg->patch_size;
    int n_heads = cfg->num_heads;
    int head_dim = model->head_dim;
    int n_reg = cfg->num_register_tokens;
    int grid_h = model->grid_h;
    int grid_w = model->grid_w;
    int n_patches = grid_h * grid_w;
    int seq_len = model->seq_len;
    int B = batch;
    float eps = 1e-5f;  /* HF DINOv3 layer_norm_eps */
    enum ggml_type act = model->use_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    struct ggml_tensor * prefix = (B > 1) ? model->token_prefix_B : model->token_prefix_1;

    /* === 1. Patch Embedding === */
    /* input: [W, H_img, 3, B], kernel: [PS, PS, 3, H] */
    /* conv2d with stride=PS, pad=0, dilation=1 */
    struct ggml_tensor * patches = ggml_conv_2d(ctx, model->patch_embed_w, input,
                                                 PS, PS, 0, 0, 1, 1);
    /* patches: [grid_w, grid_h, H, B] */

    /* Add bias: patches + bias (broadcast over spatial dims and batch) */
    struct ggml_tensor * pe_bias_4d = ggml_reshape_4d(ctx, model->patch_embed_b, 1, 1, H, 1);
    patches = ggml_add(ctx, patches, pe_bias_4d);

    /* Flatten spatial dims then transpose so channels become ne0: [H, n_patches, B] */
    patches = ggml_reshape_3d(ctx, patches, n_patches, H, B);
    patches = ggml_cont(ctx, ggml_permute(ctx, patches, 1, 0, 2, 3));

    /* Enter the activation dtype (no-op view-free cast skipped for F32) */
    if (act != GGML_TYPE_F32) {
        patches = ggml_cast(ctx, patches, act);
    }

    /* === 2. Token Assembly (HF DINOv3 order: [CLS, registers, patches]) === */
    /* CLS+register prefix is a constant [H, 1+n_reg, B] weight (identical for
     * every image), so the per-forward repeat disappears. */
    struct ggml_tensor * tokens;
    if (prefix->ne[1] > 0) {
        tokens = ggml_concat(ctx, prefix, patches, 1);
    } else {
        tokens = patches;
    }

    /* Flatten batch into the token dim: all per-layer GEMMs/norms run as one
     * large 2D op ([H, seq_len*B]) instead of B strided-batched ops, which is
     * significantly faster; only flash attention needs the batch structure. */
    tokens = ggml_reshape_2d(ctx, tokens, H, (int64_t)seq_len * B);

    /* === 3. Transformer Blocks === */
    for (int l = 0; l < cfg->num_layers; l++) {
        vit_layer_weights_t * L = &model->layers[l];
        struct ggml_tensor * residual = tokens;

        /* --- Attention sub-block --- */
        /* Plain LayerNorm (affine params folded into qkv weights/bias) */
        struct ggml_tensor * normed = ggml_norm(ctx, tokens, eps);

        /* QKV projection: [H, 3H] @ [H, seq_len*B] -> [3H, seq_len*B]
         * Each column is [q(H), k(H), v(H)] */
        struct ggml_tensor * qkv = mm_act(model, ctx, L->qkv_w, normed);
        qkv = ggml_add(ctx, qkv, L->qkv_b);

        /* Head-major views [head_dim, n_heads, seq_len, B] straight into the
         * QKV columns (rope consumes this layout directly). */
        const size_t es = ggml_type_size(qkv->type);
        size_t qkv_nb3 = qkv->nb[1] * (size_t)seq_len;
        struct ggml_tensor * Qv = ggml_view_4d(ctx, qkv, head_dim, n_heads, seq_len, B,
                                               (size_t)head_dim * es,
                                               qkv->nb[1], qkv_nb3,
                                               0);
        struct ggml_tensor * Kv = ggml_view_4d(ctx, qkv, head_dim, n_heads, seq_len, B,
                                               (size_t)head_dim * es,
                                               qkv->nb[1], qkv_nb3,
                                               (size_t)H * es);
        struct ggml_tensor * Vv = ggml_view_4d(ctx, qkv, head_dim, n_heads, seq_len, B,
                                               (size_t)head_dim * es,
                                               qkv->nb[1], qkv_nb3,
                                               (size_t)2 * H * es);

        /* DINOv3 RoPE on Q and K (identity for CLS/register rows), then plain
         * permuted VIEWS into the [head_dim, seq_len, n_heads, B] layout
         * flash attention wants — the fattn kernels read Q/K/V through their
         * row strides, so no materializing copy is needed at all.
         * Q comes out F32 (fattn requires it), K/V stay F16. */
        struct ggml_tensor * Qp = apply_rope_vision(model, ctx, Qv, GGML_TYPE_F32);
        struct ggml_tensor * Kp = apply_rope_vision(model, ctx, Kv, act);
        Qp = ggml_permute(ctx, Qp, 0, 2, 1, 3);
        Kp = ggml_permute(ctx, Kp, 0, 2, 1, 3);
        struct ggml_tensor * Vp = ggml_permute(ctx, Vv, 0, 2, 1, 3);

        /* Fused flash attention.
         * q: [head_dim, seq_len, n_heads, B], k/v likewise (V not transposed).
         * Result: [head_dim, n_heads, seq_len, B] -> reshape to [H, seq_len*B]. */
        float attn_scale = 1.0f / sqrtf((float)head_dim);
        struct ggml_tensor * attn_out = ggml_flash_attn_ext(ctx, Qp, Kp, Vp, NULL,
                                                            attn_scale, 0.0f, 0.0f);
        attn_out = ggml_reshape_2d(ctx, attn_out, H, (int64_t)seq_len * B);

        /* Output projection (ls1 LayerScale folded into weights/bias):
         * [H, H] @ [H, seq_len*B] -> [H, seq_len*B] */
        struct ggml_tensor * proj_out = mm_act(model, ctx, L->proj_w, attn_out);
        proj_out = ggml_add(ctx, proj_out, L->proj_b);

        /* Residual */
        tokens = ggml_add(ctx, residual, proj_out);

        /* --- MLP sub-block --- */
        residual = tokens;

        /* Plain LayerNorm (affine params folded into fc1 weights/bias) */
        normed = ggml_norm(ctx, tokens, eps);

        /* FC1: [H, I] @ [H, seq_len*B] -> [I, seq_len*B] */
        struct ggml_tensor * hidden = mm_act(model, ctx, L->fc1_w, normed);
        hidden = ggml_add(ctx, hidden, L->fc1_b);

        /* GELU activation (exact erf variant, matching HF "gelu") */
        hidden = ggml_gelu_erf(ctx, hidden);

        /* FC2 (ls2 LayerScale folded in): [I, H] @ [I, seq_len*B] -> [H, seq_len*B] */
        hidden = mm_act(model, ctx, L->fc2_w, hidden);
        hidden = ggml_add(ctx, hidden, L->fc2_b);

        /* Residual */
        tokens = ggml_add(ctx, residual, hidden);
    }

    /* === 4. Final LayerNorm (F32: the output must stay F32) === */
    if (act != GGML_TYPE_F32) {
        tokens = ggml_cast(ctx, tokens, GGML_TYPE_F32);
    }
    tokens = apply_layer_norm(ctx, tokens, model->norm_w, model->norm_b, eps);

    /* === 5. Output: flatten to [output_dim, B] === */
    tokens = ggml_reshape_2d(ctx, tokens, model->output_dim, B);

    /* Mark as output */
    ggml_set_output(tokens);

    return tokens;
}

/* ======================================================================== */
/* Inference                                                                 */
/* ======================================================================== */

/* Build the compute graphs (single-image and batched) and allocate them on
 * the backend. Called once at model creation; reused for every inference. */
static int build_compute_graph(ggml_vit_model_t * model) {
    size_t compute_mem = (size_t)64 * 1024 * 1024; /* metadata for graph nodes */

    /* --- Single-image graph --- */
    {
        struct ggml_init_params params = {
            .mem_size   = compute_mem,
            .mem_buffer = NULL,
            .no_alloc   = true,
        };
        model->compute_ctx = ggml_init(params);
        if (!model->compute_ctx) return -1;
        struct ggml_context * ctx = model->compute_ctx;

        model->input_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                                 model->config.input_width,
                                                 model->config.input_height, 3, 1);
        ggml_set_input(model->input_tensor);
        ggml_set_name(model->input_tensor, "input");

        model->output_tensor = build_forward(model, ctx, model->input_tensor, 1);

        model->graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(model->graph, model->output_tensor);

        if (!ggml_gallocr_alloc_graph(model->galloc, model->graph)) {
            fprintf(stderr, "ggml_vit: failed to allocate graph\n");
            return -1;
        }
    }

    /* --- Batched graph (max_batch images per forward) --- */
    if (model->max_batch > 1) {
        struct ggml_init_params params = {
            .mem_size   = compute_mem,
            .mem_buffer = NULL,
            .no_alloc   = true,
        };
        model->compute_ctx_batch = ggml_init(params);
        if (!model->compute_ctx_batch) return -1;
        struct ggml_context * ctx = model->compute_ctx_batch;

        model->input_batch = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                                model->config.input_width,
                                                model->config.input_height, 3,
                                                model->max_batch);
        ggml_set_input(model->input_batch);
        ggml_set_name(model->input_batch, "input_batch");

        model->output_batch = build_forward(model, ctx, model->input_batch, model->max_batch);

        model->graph_batch = ggml_new_graph(ctx);
        ggml_build_forward_expand(model->graph_batch, model->output_batch);

        model->galloc_batch = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model->backend));
        if (!model->galloc_batch ||
            !ggml_gallocr_alloc_graph(model->galloc_batch, model->graph_batch)) {
            fprintf(stderr, "ggml_vit: failed to allocate batched graph (max_batch=%d)\n",
                    model->max_batch);
            return -1;
        }
    }
    return 0;
}

int ggml_vit_infer(ggml_vit_model_t* model,
                   const float* input, int height, int width,
                   float* output, int output_size)
{
    if (output_size < model->output_dim) return -1;
    if (height != model->config.input_height || width != model->config.input_width) {
        fprintf(stderr, "ggml_vit: input size %dx%d does not match configured %dx%d\n",
                width, height, model->config.input_width, model->config.input_height);
        return -1;
    }

    /* Upload input into the preallocated graph input tensor */
    ggml_backend_tensor_set(model->input_tensor, input, 0, ggml_nbytes(model->input_tensor));

    /* Compute (same graph every call -> CUDA graph friendly) */
    if (ggml_backend_graph_compute(model->backend, model->graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ggml_vit: graph compute failed\n");
        return -1;
    }

    /* Download output */
    ggml_backend_tensor_get(model->output_tensor, output, 0, (size_t)model->output_dim * sizeof(float));

    return 0;
}

/* Shared batched-forward worker: runs the batched graph over the input in
 * max_batch slices. cls_only=true downloads just the leading hidden_size
 * floats (CLS token) of each image into output (n_images * hidden_size
 * floats); false downloads full output_dim vectors. */
static int infer_batch_impl(ggml_vit_model_t* model,
                            const float* input, int n_images, int height, int width,
                            float* output, int output_size, int cls_only) {
    if (n_images <= 0) return -1;
    if (height != model->config.input_height || width != model->config.input_width) {
        fprintf(stderr, "ggml_vit: input size %dx%d does not match configured %dx%d\n",
                width, height, model->config.input_width, model->config.input_height);
        return -1;
    }
    const int out_vec = cls_only ? model->config.hidden_size : model->output_dim;
    if ((size_t)output_size < (size_t)n_images * out_vec) return -1;

    const size_t img_floats = (size_t)3 * height * width;
    const int B = model->max_batch;

    int done = 0;
    while (done < n_images) {
        int rem = n_images - done;
        if (rem == 1 || B <= 1) {
            if (cls_only) {
                /* Single-image graph writes the full vector: stage it and
                 * copy the leading CLS floats into the compact output. */
                float * tmp = (float*)malloc((size_t)model->output_dim * sizeof(float));
                if (!tmp) return -1;
                int rc = ggml_vit_infer(model, input + (size_t)done * img_floats, height, width,
                                        tmp, model->output_dim);
                if (rc == 0) {
                    memcpy(output + (size_t)done * out_vec, tmp,
                           (size_t)out_vec * sizeof(float));
                }
                free(tmp);
                if (rc != 0) return -1;
            } else {
                if (ggml_vit_infer(model, input + (size_t)done * img_floats, height, width,
                                   output + (size_t)done * out_vec, model->output_dim) != 0) {
                    return -1;
                }
            }
            done += 1;
            continue;
        }

        int take = rem < B ? rem : B;
        /* Upload `take` images; slots beyond `take` keep stale data and their
         * results are simply discarded. Graph topology stays fixed at B. */
        ggml_backend_tensor_set(model->input_batch,
                                input + (size_t)done * img_floats,
                                0, (size_t)take * img_floats * sizeof(float));

        if (ggml_backend_graph_compute(model->backend, model->graph_batch) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "ggml_vit: batched graph compute failed\n");
            return -1;
        }

        if (cls_only) {
            /* CLS of image i sits at byte offset i*output_dim in the
             * [output_dim, B] output tensor: one tiny device read per image. */
            for (int i = 0; i < take; i++) {
                ggml_backend_tensor_get(model->output_batch,
                                        output + (size_t)(done + i) * out_vec,
                                        (size_t)i * model->output_dim * sizeof(float),
                                        (size_t)out_vec * sizeof(float));
            }
        } else {
            ggml_backend_tensor_get(model->output_batch,
                                    output + (size_t)done * out_vec,
                                    0, (size_t)take * model->output_dim * sizeof(float));
        }
        done += take;
    }
    return 0;
}

int ggml_vit_infer_batch(ggml_vit_model_t* model,
                         const float* input, int n_images, int height, int width,
                         float* output, int output_size)
{
    return infer_batch_impl(model, input, n_images, height, width, output, output_size, 0);
}

int ggml_vit_infer_batch_cls(ggml_vit_model_t* model,
                             const float* input, int n_images, int height, int width,
                             float* output, int output_size)
{
    return infer_batch_impl(model, input, n_images, height, width, output, output_size, 1);
}

/* Profiling helper: replay the prebuilt graph `iters` times with no input
 * upload or output download (input tensor keeps its last contents).
 * Returns seconds per replay, or -1 on failure. Syncs before and after.
 * Runs 10 unmeasured warmup iterations first so one-time costs (CUDA graph
 * instantiate, re-capture after gallocr realloc) stay out of the average. */
double ggml_vit_bench_graph(ggml_vit_model_t* model, int iters)
{
    struct ggml_cgraph * g = model->max_batch > 1 ? model->graph_batch : model->graph;
    if (!g || iters <= 0) return -1.0;
    for (int i = 0; i < 10; i++) {
        if (ggml_backend_graph_compute(model->backend, g) != GGML_STATUS_SUCCESS) return -1.0;
    }
    ggml_backend_synchronize(model->backend);
#ifdef _WIN32
    LARGE_INTEGER f, t0, t1;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t0);
#else
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
#endif
    for (int i = 0; i < iters; i++) {
        if (ggml_backend_graph_compute(model->backend, g) != GGML_STATUS_SUCCESS) return -1.0;
    }
    ggml_backend_synchronize(model->backend);
#ifdef _WIN32
    QueryPerformanceCounter(&t1);
    return (double)(t1.QuadPart - t0.QuadPart) / (double)f.QuadPart / (double)iters;
#else
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    return (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
#endif
}

/* ======================================================================== */
/* Cleanup                                                                   */
/* ======================================================================== */

int ggml_vit_get_output_size(const ggml_vit_model_t* model) {
    return model->output_dim;
}

void ggml_vit_destroy(ggml_vit_model_t* model) {
    if (!model) return;
    if (model->compute_ctx_batch) ggml_free(model->compute_ctx_batch);
    if (model->galloc_batch) ggml_gallocr_free(model->galloc_batch);
    if (model->compute_ctx) ggml_free(model->compute_ctx);
    if (model->galloc) ggml_gallocr_free(model->galloc);
    if (model->weight_buf) ggml_backend_buffer_free(model->weight_buf);
    if (model->weight_ctx) ggml_free(model->weight_ctx);
    if (model->backend) ggml_backend_free(model->backend);
    free(model->layers);
    free(model);
}

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

    /* Weight context and buffer */
    struct ggml_context * weight_ctx;
    struct ggml_backend_buffer * weight_buf;

    /* Global weights */
    struct ggml_tensor * patch_embed_w;  /* [PS, PS, 3, H] */
    struct ggml_tensor * patch_embed_b;  /* [H] */
    struct ggml_tensor * cls_token;      /* [1, 1, 1, H] */
    struct ggml_tensor * register_tok;   /* [1, 1, n_reg, H] */
    struct ggml_tensor * norm_w;         /* [H] */
    struct ggml_tensor * norm_b;         /* [H] */

    /* Per-layer weights */
    vit_layer_weights_t * layers;

    /* DINOv3 RoPE tables [head_dim, 1, seq_len], identity rows for CLS/register tokens */
    struct ggml_tensor * rope_cos;
    struct ggml_tensor * rope_sin;

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
    struct ggml_context * ctx = model->weight_ctx;

    /* Create weight tensors. Large matmul/conv weights are stored as F16:
     * halves bandwidth and enables tensor-core GEMM on CUDA. */
    model->patch_embed_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, PS, PS, 3, H);
    model->patch_embed_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    ggml_set_name(model->patch_embed_w, "patch_embed_w");
    ggml_set_name(model->patch_embed_b, "patch_embed_b");

    model->cls_token = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 1, H);
    model->register_tok = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, n_reg, H);
    ggml_set_name(model->cls_token, "cls_token");
    ggml_set_name(model->register_tok, "register_tok");

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
        L->qkv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3 * H);
        /* Proj weight transposed: [H, H] */
        L->proj_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, H, H);
        L->proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        /* FC1 weight transposed: [H, I] */
        L->fc1_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, H, I);
        L->fc1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, I);
        /* FC2 weight transposed: [I, H] */
        L->fc2_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, I, H);
        L->fc2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);

        snprintf(name, sizeof(name), "layer_%d_qkv_w", l);  ggml_set_name(L->qkv_w, name);
        snprintf(name, sizeof(name), "layer_%d_proj_w", l); ggml_set_name(L->proj_w, name);
        snprintf(name, sizeof(name), "layer_%d_fc1_w", l);  ggml_set_name(L->fc1_w, name);
        snprintf(name, sizeof(name), "layer_%d_fc2_w", l);  ggml_set_name(L->fc2_w, name);
    }

    /* DINOv3 RoPE cos/sin tables: [head_dim, seq_len, 1], broadcast over the
     * head dim of [head_dim, seq_len, n_heads] activations. The rotate-half
     * sign is pre-folded into the sin table (first half negated). */
    model->rope_cos = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, model->seq_len, 1);
    model->rope_sin = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, model->seq_len, 1);
    ggml_set_name(model->rope_cos, "rope_cos");
    ggml_set_name(model->rope_sin, "rope_sin");

    /* Allocate weight tensors on the backend */
    model->weight_buf = ggml_backend_alloc_ctx_tensors(ctx, model->backend);
    if (!model->weight_buf) { ggml_vit_destroy(model); return NULL; }

    /* Fill DINOv3 RoPE tables (after ggml_backend_alloc_ctx_tensors).
     * Patch center coords normalized to [-1, +1], angle = 2*pi * coord * inv_freq with
     * inv_freq[j] = base^(-4j/head_dim), angle layout [y(D/4), x(D/4)] tiled twice.
     * Token order matches HF DINOv3: [CLS, registers, patches].
     * CLS and register tokens are not rotated (cos = 1, sin = 0).
     * The rotate-half sign is folded into the sin table: out = x*cos + roll(x, D/2)*sin'
     * with sin'[0:D/2] = -sin, sin'[D/2:D] = +sin.
     * Built on the host, then uploaded (backend buffer may be device memory). */
    {
        size_t n_vals = (size_t)head_dim * model->seq_len;
        float * cos_data = (float *)malloc(n_vals * sizeof(float));
        float * sin_data = (float *)malloc(n_vals * sizeof(float));
        if (!cos_data || !sin_data) {
            free(cos_data); free(sin_data);
            ggml_vit_destroy(model);
            return NULL;
        }
        const float two_pi = 6.28318530717958647692f;
        const int D = head_dim;
        const int n_prefix = config->has_cls_token + n_reg;
        for (int t = 0; t < n_prefix; t++) {
            for (int d = 0; d < D; d++) {
                cos_data[(size_t)t * D + d] = 1.0f;
                sin_data[(size_t)t * D + d] = 0.0f;
            }
        }
        for (int row = 0; row < model->grid_h; row++) {
            for (int col = 0; col < model->grid_w; col++) {
                int t = n_prefix + row * model->grid_w + col;
                float cy = 2.0f * (row + 0.5f) / model->grid_h - 1.0f;
                float cx = 2.0f * (col + 0.5f) / model->grid_w - 1.0f;
                float * c = cos_data + (size_t)t * D;
                float * s = sin_data + (size_t)t * D;
                for (int j = 0; j < D / 4; j++) {
                    float inv_freq = powf(config->rope_freq_base, -4.0f * j / D);
                    float ay = two_pi * cy * inv_freq;
                    float ax = two_pi * cx * inv_freq;
                    /* first half: sign folded in (-sin) */
                    c[j]             = cosf(ay); s[j]             = -sinf(ay);
                    c[D / 4 + j]     = cosf(ax); s[D / 4 + j]     = -sinf(ax);
                    c[D / 2 + j]     = cosf(ay); s[D / 2 + j]     = sinf(ay);
                    c[3 * D / 4 + j] = cosf(ax); s[3 * D / 4 + j] = sinf(ax);
                }
            }
        }
        ggml_backend_tensor_set(model->rope_cos, cos_data, 0, n_vals * sizeof(float));
        ggml_backend_tensor_set(model->rope_sin, sin_data, 0, n_vals * sizeof(float));
        free(cos_data);
        free(sin_data);
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
    if (read_tensor(f, model->cls_token) != 0) goto fail;
    if (read_tensor(f, model->register_tok) != 0) goto fail;

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

/* Apply LayerNorm: norm(x, eps) * w + b */
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

/* Apply DINOv3 RoPE: x*cos + roll(x, D/2)*sin_signed
 * x: [head_dim, seq_len, n_heads], cos/sin: [head_dim, seq_len, 1] (broadcast
 * over heads). The rotate-half sign is pre-folded into the sin table. */
static struct ggml_tensor * apply_rope(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * cos_t,
    struct ggml_tensor * sin_t)
{
    const int64_t D = x->ne[0];
    struct ggml_tensor * rot = ggml_roll(ctx, x, (int)(D / 2), 0, 0, 0);
    return ggml_add(ctx,
                    ggml_mul(ctx, x, cos_t),
                    ggml_mul(ctx, rot, sin_t));
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

    /* === 2. Token Assembly (HF DINOv3 order: [CLS, registers, patches]) === */
    /* CLS token: [H, 1] -> broadcast to [H, 1, B] */
    struct ggml_tensor * tokens = ggml_reshape_2d(ctx, model->cls_token, H, 1);
    if (B > 1) tokens = ggml_repeat_4d(ctx, tokens, H, 1, B, 1);

    /* Concatenate registers: [H, 1 + n_reg, B] */
    if (n_reg > 0) {
        struct ggml_tensor * regs = ggml_reshape_2d(ctx, model->register_tok, H, n_reg);
        if (B > 1) regs = ggml_repeat_4d(ctx, regs, H, n_reg, B, 1);
        tokens = ggml_concat(ctx, tokens, regs, 1);
    }

    /* Concatenate patches: [H, seq_len, B] */
    tokens = ggml_concat(ctx, tokens, patches, 1);

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
        struct ggml_tensor * qkv = ggml_mul_mat(ctx, L->qkv_w, normed);
        qkv = ggml_add(ctx, qkv, L->qkv_b);

        /* Split into Q, K, V head-major views and permute to
         * [head_dim, seq_len, n_heads, B] in a single cont each.
         * Within a qkv column: head h of Q at offset h*head_dim, K at H, V at 2H. */
        size_t qkv_nb3 = qkv->nb[1] * (size_t)seq_len;
        struct ggml_tensor * Qv = ggml_view_4d(ctx, qkv, head_dim, n_heads, seq_len, B,
                                               (size_t)head_dim * sizeof(float),
                                               qkv->nb[1], qkv_nb3,
                                               0);
        struct ggml_tensor * Kv = ggml_view_4d(ctx, qkv, head_dim, n_heads, seq_len, B,
                                               (size_t)head_dim * sizeof(float),
                                               qkv->nb[1], qkv_nb3,
                                               (size_t)H * sizeof(float));
        struct ggml_tensor * Vv = ggml_view_4d(ctx, qkv, head_dim, n_heads, seq_len, B,
                                               (size_t)head_dim * sizeof(float),
                                               qkv->nb[1], qkv_nb3,
                                               (size_t)2 * H * sizeof(float));

        struct ggml_tensor * Qp = ggml_cont(ctx, ggml_permute(ctx, Qv, 0, 2, 1, 3));
        struct ggml_tensor * Kp = ggml_cont(ctx, ggml_permute(ctx, Kv, 0, 2, 1, 3));

        /* Apply DINOv3 RoPE to Q and K (identity for CLS/register rows) */
        Qp = apply_rope(ctx, Qp, model->rope_cos, model->rope_sin);
        Kp = apply_rope(ctx, Kp, model->rope_cos, model->rope_sin);

        /* Fused flash attention.
         * q: [head_dim, seq_len, n_heads, B], k/v likewise (V not transposed),
         * K/V cast to F16 for the fattn kernels.
         * Result: [head_dim, n_heads, seq_len, B] -> reshape to [H, seq_len*B]. */
        struct ggml_tensor * Kp16 = ggml_cast(ctx, Kp, GGML_TYPE_F16);
        struct ggml_tensor * Vp16 = ggml_cast(ctx, ggml_permute(ctx, Vv, 0, 2, 1, 3), GGML_TYPE_F16);

        float attn_scale = 1.0f / sqrtf((float)head_dim);
        struct ggml_tensor * attn_out = ggml_flash_attn_ext(ctx, Qp, Kp16, Vp16, NULL,
                                                            attn_scale, 0.0f, 0.0f);
        attn_out = ggml_reshape_2d(ctx, attn_out, H, (int64_t)seq_len * B);

        /* Output projection (ls1 LayerScale folded into weights/bias):
         * [H, H] @ [H, seq_len*B] -> [H, seq_len*B] */
        struct ggml_tensor * proj_out = ggml_mul_mat(ctx, L->proj_w, attn_out);
        proj_out = ggml_add(ctx, proj_out, L->proj_b);

        /* Residual */
        tokens = ggml_add(ctx, residual, proj_out);

        /* --- MLP sub-block --- */
        residual = tokens;

        /* Plain LayerNorm (affine params folded into fc1 weights/bias) */
        normed = ggml_norm(ctx, tokens, eps);

        /* FC1: [H, I] @ [H, seq_len*B] -> [I, seq_len*B] */
        struct ggml_tensor * hidden = ggml_mul_mat(ctx, L->fc1_w, normed);
        hidden = ggml_add(ctx, hidden, L->fc1_b);

        /* GELU activation (exact erf variant, matching HF "gelu") */
        hidden = ggml_gelu_erf(ctx, hidden);

        /* FC2 (ls2 LayerScale folded in): [I, H] @ [I, seq_len*B] -> [H, seq_len*B] */
        hidden = ggml_mul_mat(ctx, L->fc2_w, hidden);
        hidden = ggml_add(ctx, hidden, L->fc2_b);

        /* Residual */
        tokens = ggml_add(ctx, residual, hidden);
    }

    /* === 4. Final LayerNorm === */
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

int ggml_vit_infer_batch(ggml_vit_model_t* model,
                         const float* input, int n_images, int height, int width,
                         float* output, int output_size)
{
    if (n_images <= 0) return -1;
    if (height != model->config.input_height || width != model->config.input_width) {
        fprintf(stderr, "ggml_vit: input size %dx%d does not match configured %dx%d\n",
                width, height, model->config.input_width, model->config.input_height);
        return -1;
    }
    if ((size_t)output_size < (size_t)n_images * model->output_dim) return -1;

    const size_t img_floats = (size_t)3 * height * width;
    const size_t out_floats = (size_t)model->output_dim;
    const int B = model->max_batch;

    int done = 0;
    while (done < n_images) {
        int rem = n_images - done;
        if (rem == 1 || B <= 1) {
            if (ggml_vit_infer(model, input + (size_t)done * img_floats, height, width,
                               output + (size_t)done * out_floats, model->output_dim) != 0) {
                return -1;
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

        ggml_backend_tensor_get(model->output_batch,
                                output + (size_t)done * out_floats,
                                0, (size_t)take * out_floats * sizeof(float));
        done += take;
    }
    return 0;
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

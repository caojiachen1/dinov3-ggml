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

/* Per-layer weight tensors */
typedef struct {
    struct ggml_tensor * norm1_w;      /* [H] */
    struct ggml_tensor * norm1_b;      /* [H] */
    struct ggml_tensor * qkv_w;        /* [H, 3*H] */
    struct ggml_tensor * qkv_b;        /* [3*H] */
    struct ggml_tensor * proj_w;       /* [H, H] */
    struct ggml_tensor * proj_b;       /* [H] */
    struct ggml_tensor * norm2_w;      /* [H] */
    struct ggml_tensor * norm2_b;      /* [H] */
    struct ggml_tensor * fc1_w;        /* [H, I] */
    struct ggml_tensor * fc1_b;        /* [I] */
    struct ggml_tensor * fc2_w;        /* [I, H] */
    struct ggml_tensor * fc2_b;        /* [H] */
    struct ggml_tensor * ls1_gamma;    /* [H] LayerScale for attention */
    struct ggml_tensor * ls2_gamma;    /* [H] LayerScale for MLP */
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
};

/* Helper: read a full tensor from file and upload it to the backend buffer.
 * Works for both CPU and GPU (device) buffers. */
static int read_tensor(FILE * f, struct ggml_tensor * t) {
    size_t n = (size_t)ggml_nelements(t);
    float * buf = (float *)malloc(n * sizeof(float));
    if (!buf) return -1;
    if (fread(buf, sizeof(float), n, f) != n) { free(buf); return -1; }
    ggml_backend_tensor_set(t, buf, 0, n * sizeof(float));
    free(buf);
    return 0;
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

    /* Create weight tensors */
    model->patch_embed_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, PS, PS, 3, H);
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

    /* Per-layer weights */
    for (int l = 0; l < config->num_layers; l++) {
        vit_layer_weights_t * L = &model->layers[l];
        char name[64];

        L->norm1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        L->norm1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        /* QKV weight transposed for ggml_mul_mat: [H, 3*H] */
        L->qkv_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 3 * H);
        L->qkv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3 * H);
        /* Proj weight transposed: [H, H] */
        L->proj_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, H);
        L->proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        L->norm2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        L->norm2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        /* FC1 weight transposed: [H, I] */
        L->fc1_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, I);
        L->fc1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, I);
        /* FC2 weight transposed: [I, H] */
        L->fc2_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, I, H);
        L->fc2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        L->ls1_gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        L->ls2_gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);

        snprintf(name, sizeof(name), "layer_%d_norm1_w", l); ggml_set_name(L->norm1_w, name);
        snprintf(name, sizeof(name), "layer_%d_qkv_w", l);  ggml_set_name(L->qkv_w, name);
        snprintf(name, sizeof(name), "layer_%d_proj_w", l); ggml_set_name(L->proj_w, name);
        snprintf(name, sizeof(name), "layer_%d_fc1_w", l);  ggml_set_name(L->fc1_w, name);
        snprintf(name, sizeof(name), "layer_%d_fc2_w", l);  ggml_set_name(L->fc2_w, name);
    }

    /* DINOv3 RoPE cos/sin tables, broadcast over heads: [head_dim, 1, seq_len] */
    model->rope_cos = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, 1, model->seq_len);
    model->rope_sin = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, 1, model->seq_len);
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
                    c[j]             = cosf(ay); s[j]             = sinf(ay);
                    c[D / 4 + j]     = cosf(ax); s[D / 4 + j]     = sinf(ax);
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

    return model;
}

/* ======================================================================== */
/* Weight loading                                                            */
/* ======================================================================== */

int ggml_vit_load_weights(ggml_vit_model_t* model, const char* path) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ggml_vit: cannot open %s\n", path); return -1; }

    int n_layers = model->config.num_layers;

    /* All 2D/4D weights are read as-is: PyTorch row-major [out, in] (or [OC,IC,KH,KW])
     * memory layout matches the GGML ne0-contiguous layout of the created tensors. */
    if (read_tensor(f, model->patch_embed_w) != 0) goto fail;
    if (read_tensor(f, model->patch_embed_b) != 0) goto fail;
    if (read_tensor(f, model->cls_token) != 0) goto fail;
    if (read_tensor(f, model->register_tok) != 0) goto fail;

    /* Per-layer weights */
    for (int l = 0; l < n_layers; l++) {
        vit_layer_weights_t * L = &model->layers[l];

        if (read_tensor(f, L->norm1_w) != 0) goto fail;
        if (read_tensor(f, L->norm1_b) != 0) goto fail;
        if (read_tensor(f, L->qkv_w) != 0) goto fail;
        if (read_tensor(f, L->qkv_b) != 0) goto fail;
        if (read_tensor(f, L->proj_w) != 0) goto fail;
        if (read_tensor(f, L->proj_b) != 0) goto fail;
        if (read_tensor(f, L->norm2_w) != 0) goto fail;
        if (read_tensor(f, L->norm2_b) != 0) goto fail;
        if (read_tensor(f, L->fc1_w) != 0) goto fail;
        if (read_tensor(f, L->fc1_b) != 0) goto fail;
        if (read_tensor(f, L->fc2_w) != 0) goto fail;
        if (read_tensor(f, L->fc2_b) != 0) goto fail;
        if (read_tensor(f, L->ls1_gamma) != 0) goto fail;
        if (read_tensor(f, L->ls2_gamma) != 0) goto fail;
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

/* Apply LayerScale: x * gamma where gamma is [H] and x is [H, seq_len] */
static struct ggml_tensor * apply_layer_scale(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * gamma)
{
    /* gamma is [H], x is [H, seq_len] -> direct broadcast mul */
    return ggml_mul(ctx, x, gamma);
}

/* Apply DINOv3 RoPE: x*cos + rotate_half(x)*sin
 * x: [head_dim, n_heads, seq_len], cos/sin: [head_dim, 1, seq_len] (broadcast over heads)
 * rotate_half([x1, x2]) = [-x2, x1] with halves split along dim 0 */
static struct ggml_tensor * apply_rope(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * cos_t,
    struct ggml_tensor * sin_t)
{
    const int64_t D = x->ne[0];
    const int64_t n_heads = x->ne[1];
    const int64_t seq_len = x->ne[2];
    struct ggml_tensor * x1 = ggml_view_3d(ctx, x, D / 2, n_heads, seq_len,
                                            x->nb[1], x->nb[2], 0);
    struct ggml_tensor * x2 = ggml_view_3d(ctx, x, D / 2, n_heads, seq_len,
                                            x->nb[1], x->nb[2], (D / 2) * sizeof(float));
    struct ggml_tensor * rot = ggml_concat(ctx,
                                            ggml_neg(ctx, ggml_cont(ctx, x2)),
                                            ggml_cont(ctx, x1), 0);
    return ggml_add(ctx,
                    ggml_mul(ctx, x, cos_t),
                    ggml_mul(ctx, rot, sin_t));
}

static struct ggml_tensor * build_forward(
    ggml_vit_model_t * model,
    struct ggml_context * ctx,
    struct ggml_tensor * input)
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
    float eps = 1e-5f;  /* HF DINOv3 layer_norm_eps */

    /* === 1. Patch Embedding === */
    /* input: [W, H_img, 3, 1], kernel: [PS, PS, 3, H] */
    /* conv2d with stride=PS, pad=0, dilation=1 */
    struct ggml_tensor * patches = ggml_conv_2d(ctx, model->patch_embed_w, input,
                                                 PS, PS, 0, 0, 1, 1);
    /* patches: [grid_w, grid_h, H, 1] */

    /* Add bias: patches + bias (broadcast over spatial dims) */
    /* conv2d output is [grid_w, grid_h, H, 1], bias must be [1, 1, H, 1] to broadcast */
    struct ggml_tensor * pe_bias_4d = ggml_reshape_4d(ctx, model->patch_embed_b, 1, 1, H, 1);
    patches = ggml_add(ctx, patches, pe_bias_4d);

    /* Flatten spatial dims then transpose so channels become ne0: [H, n_patches] */
    patches = ggml_reshape_2d(ctx, patches, n_patches, H);
    patches = ggml_cont(ctx, ggml_transpose(ctx, patches));

    /* === 2. Token Assembly (HF DINOv3 order: [CLS, registers, patches]) === */
    /* CLS token: [H, 1] */
    struct ggml_tensor * tokens = ggml_reshape_2d(ctx, model->cls_token, H, 1);

    /* Concatenate registers: [H, 1 + n_reg] */
    if (n_reg > 0) {
        struct ggml_tensor * regs = ggml_reshape_2d(ctx, model->register_tok, H, n_reg);
        tokens = ggml_concat(ctx, tokens, regs, 1);
    }

    /* Concatenate patches: [H, seq_len] */
    tokens = ggml_concat(ctx, tokens, patches, 1);

    /* === 3. Transformer Blocks === */
    for (int l = 0; l < cfg->num_layers; l++) {
        vit_layer_weights_t * L = &model->layers[l];
        struct ggml_tensor * residual = tokens;

        /* --- Attention sub-block --- */
        /* LayerNorm */
        struct ggml_tensor * normed = apply_layer_norm(ctx, tokens, L->norm1_w, L->norm1_b, eps);

        /* QKV projection: [H, 3H] @ [H, seq_len] -> [3H, seq_len]
         * Each column is [q(H), k(H), v(H)] */
        struct ggml_tensor * qkv = ggml_mul_mat(ctx, L->qkv_w, normed);
        qkv = ggml_add(ctx, qkv, L->qkv_b);

        /* Split into Q, K, V heads: [head_dim, n_heads, seq_len]
         * Within a column: head h of Q at offset h*head_dim, K starts at H, V at 2H */
        struct ggml_tensor * Q = ggml_cont(ctx, ggml_view_3d(ctx, qkv, head_dim, n_heads, seq_len,
                                               (size_t)head_dim * sizeof(float),
                                               qkv->nb[1],
                                               0));
        struct ggml_tensor * K = ggml_cont(ctx, ggml_view_3d(ctx, qkv, head_dim, n_heads, seq_len,
                                               (size_t)head_dim * sizeof(float),
                                               qkv->nb[1],
                                               (size_t)H * sizeof(float)));
        struct ggml_tensor * V = ggml_cont(ctx, ggml_view_3d(ctx, qkv, head_dim, n_heads, seq_len,
                                               (size_t)head_dim * sizeof(float),
                                               qkv->nb[1],
                                               (size_t)2 * H * sizeof(float)));

        /* Apply DINOv3 RoPE to Q and K (identity for CLS/register rows) */
        Q = apply_rope(ctx, Q, model->rope_cos, model->rope_sin);
        K = apply_rope(ctx, K, model->rope_cos, model->rope_sin);

        /* Standard scaled dot-product attention per head */
        /* [head_dim, n_heads, seq_len] -> [head_dim, seq_len, n_heads] */
        struct ggml_tensor * Qp = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        struct ggml_tensor * Kp = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));

        /* Scores: [seq_len(k), seq_len(q), n_heads] */
        struct ggml_tensor * KQ = ggml_mul_mat(ctx, Kp, Qp);
        float attn_scale = 1.0f / sqrtf((float)head_dim);
        KQ = ggml_soft_max_ext(ctx, KQ, NULL, attn_scale, 0.0f);

        /* V^T: [seq_len, head_dim, n_heads] */
        struct ggml_tensor * Vt = ggml_cont(ctx, ggml_permute(ctx, V, 1, 2, 0, 3));

        /* KQV: [head_dim, seq_len, n_heads] */
        struct ggml_tensor * KQV = ggml_mul_mat(ctx, Vt, KQ);

        /* Merge heads: [head_dim, n_heads, seq_len] -> [H, seq_len] */
        struct ggml_tensor * attn_out = ggml_cont(ctx, ggml_permute(ctx, KQV, 0, 2, 1, 3));
        attn_out = ggml_reshape_2d(ctx, attn_out, H, seq_len);

        /* Output projection: [H, H] @ [H, seq_len] -> [H, seq_len] */
        struct ggml_tensor * proj_out = ggml_mul_mat(ctx, L->proj_w, attn_out);
        proj_out = ggml_add(ctx, proj_out, L->proj_b);

        /* LayerScale */
        proj_out = apply_layer_scale(ctx, proj_out, L->ls1_gamma);

        /* Residual */
        tokens = ggml_add(ctx, residual, proj_out);

        /* --- MLP sub-block --- */
        residual = tokens;

        /* LayerNorm */
        normed = apply_layer_norm(ctx, tokens, L->norm2_w, L->norm2_b, eps);

        /* FC1: [H, I] @ [H, seq_len] -> [I, seq_len] */
        struct ggml_tensor * hidden = ggml_mul_mat(ctx, L->fc1_w, normed);
        hidden = ggml_add(ctx, hidden, L->fc1_b);

        /* GELU activation (exact erf variant, matching HF "gelu") */
        hidden = ggml_gelu_erf(ctx, hidden);

        /* FC2: [I, H] @ [I, seq_len] -> [H, seq_len] */
        hidden = ggml_mul_mat(ctx, L->fc2_w, hidden);
        hidden = ggml_add(ctx, hidden, L->fc2_b);

        /* LayerScale */
        hidden = apply_layer_scale(ctx, hidden, L->ls2_gamma);

        /* Residual */
        tokens = ggml_add(ctx, residual, hidden);
    }

    /* === 4. Final LayerNorm === */
    tokens = apply_layer_norm(ctx, tokens, model->norm_w, model->norm_b, eps);

    /* === 5. Output: flatten to 1D === */
    tokens = ggml_reshape_1d(ctx, tokens, model->output_dim);

    /* Mark as output */
    ggml_set_output(tokens);

    return tokens;
}

/* ======================================================================== */
/* Inference                                                                 */
/* ======================================================================== */

int ggml_vit_infer(ggml_vit_model_t* model,
                   const float* input, int height, int width,
                   float* output, int output_size)
{
    if (output_size < model->output_dim) return -1;

    int H = model->config.hidden_size;
    int img_h = height;
    int img_w = width;

    /* Compute context (no_alloc = true, graph allocator handles memory) */
    size_t compute_mem = (size_t)512 * 1024 * 1024; /* 512 MB for compute graph */
    struct ggml_init_params params = {
        .mem_size   = compute_mem,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) return -1;

    /* Create input tensor: [width, height, 3, 1] */
    struct ggml_tensor * input_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, img_w, img_h, 3, 1);
    ggml_set_input(input_tensor);
    ggml_set_name(input_tensor, "input");

    /* Build computation graph */
    struct ggml_tensor * output_tensor = build_forward(model, ctx, input_tensor);

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output_tensor);

    /* Allocate graph via galloc */
    if (!ggml_gallocr_alloc_graph(model->galloc, graph)) {
        fprintf(stderr, "ggml_vit: failed to allocate graph\n");
        ggml_free(ctx);
        return -1;
    }

    /* Copy input data */
    ggml_backend_tensor_set(input_tensor, input, 0, ggml_nbytes(input_tensor));

    /* Compute */
    if (ggml_backend_graph_compute(model->backend, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ggml_vit: graph compute failed\n");
        ggml_free(ctx);
        return -1;
    }

    /* Get output */
    ggml_backend_tensor_get(output_tensor, output, 0, (size_t)model->output_dim * sizeof(float));

    ggml_free(ctx);
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
    if (model->galloc) ggml_gallocr_free(model->galloc);
    if (model->weight_buf) ggml_backend_buffer_free(model->weight_buf);
    if (model->weight_ctx) ggml_free(model->weight_ctx);
    if (model->backend) ggml_backend_free(model->backend);
    free(model->layers);
    free(model);
}

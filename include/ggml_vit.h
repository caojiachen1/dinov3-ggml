#ifndef GGML_VIT_H
#define GGML_VIT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Model configuration */
typedef struct {
    int hidden_size;          /* e.g. 384 for ViT-S */
    int num_layers;           /* e.g. 12 for ViT-S */
    int num_heads;            /* e.g. 6 for ViT-S */
    int intermediate_size;    /* e.g. 1536 for ViT-S */
    int patch_size;           /* e.g. 16 */
    int num_register_tokens;  /* e.g. 4 */
    int input_height;         /* e.g. 518 */
    int input_width;          /* e.g. 518 */
    int has_cls_token;        /* 1 = include CLS token */
    float layer_scale_eps;    /* LayerScale initial value, typically 1.0 */
    /* RoPE config */
    float rope_freq_base;     /* DINOv3 rope_theta, typically 100.0 */
    /* Max images per batched forward pass (0 or 1 = single-image graphs only) */
    int max_batch;
} vit_config_t;

/* Opaque model handle */
typedef struct ggml_vit_model ggml_vit_model_t;

/* Create model with given configuration.
 * Returns NULL on failure. */
ggml_vit_model_t* ggml_vit_create(const vit_config_t* config);

/* Load weights from raw binary file.
 * Format: sequential f32 arrays for each weight in order:
 *   patch_embed_weight [hidden_size * 3 * patch_size * patch_size]
 *   patch_embed_bias   [hidden_size]
 *   cls_token          [hidden_size]
 *   register_tokens    [num_register_tokens * hidden_size]
 *   For each layer:
 *     norm1_weight     [hidden_size]
 *     norm1_bias       [hidden_size]
 *     qkv_weight       [3 * hidden_size * hidden_size]
 *     qkv_bias         [3 * hidden_size]
 *     proj_weight      [hidden_size * hidden_size]
 *     proj_bias        [hidden_size]
 *     norm2_weight     [hidden_size]
 *     norm2_bias       [hidden_size]
 *     fc1_weight       [intermediate_size * hidden_size]
 *     fc1_bias        [intermediate_size]
 *     fc2_weight       [hidden_size * intermediate_size]
 *     fc2_bias        [hidden_size]
 *     ls1_gamma        [hidden_size]  (LayerScale)
 *     ls2_gamma        [hidden_size]  (LayerScale)
 *   norm_weight        [hidden_size]
 *   norm_bias          [hidden_size]
 * Returns 0 on success, -1 on failure. */
int ggml_vit_load_weights(ggml_vit_model_t* model, const char* path);

/* Run inference on preprocessed input image.
 * input: NCHW float tensor [1, 3, height, width]
 * output: feature vector buffer
 * output_size: size of output buffer in floats
 * Returns 0 on success, -1 on failure. */
int ggml_vit_infer(ggml_vit_model_t* model,
                   const float* input, int height, int width,
                   float* output, int output_size);

/* Run batched inference on n_images preprocessed images stored contiguously
 * (each 3*height*width floats). Output receives n_images feature vectors
 * back to back. output_size is the total buffer size in floats.
 * Returns 0 on success, -1 on failure. */
int ggml_vit_infer_batch(ggml_vit_model_t* model,
                         const float* input, int n_images, int height, int width,
                         float* output, int output_size);

/* Free model and all associated resources. */
void ggml_vit_destroy(ggml_vit_model_t* model);

/* Get the output feature dimension.
 * This is: sequence_length * hidden_size
 * where sequence_length = 1 (CLS) + num_patches + num_register_tokens */
int ggml_vit_get_output_size(const ggml_vit_model_t* model);

#ifdef __cplusplus
}
#endif

#endif /* GGML_VIT_H */

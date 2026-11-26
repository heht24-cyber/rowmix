#ifndef ROWMIX_H
#define ROWMIX_H
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
typedef struct rm_model rm_model;
typedef void (*rm_trace_fn)(uint32_t node, const int8_t *values, size_t count, void *user);

/* File data is copied and validated during loading. Call rm_close once. */
rm_model *rm_open_file(const char *path, char *error, size_t error_size);
enum rm_kernel { RM_AUTO=0, RM_SCALAR=1, RM_AVX2=2 };
rm_model *rm_open_file_ex(const char *path, enum rm_kernel kernel, char *error, size_t error_size);
const char *rm_backend(const rm_model *model);
void rm_close(rm_model *model);
size_t rm_inputs(const rm_model *model);
size_t rm_outputs(const rm_model *model);
size_t rm_arena_bytes(const rm_model *model);
size_t rm_node_count(const rm_model *model);
size_t rm_node_elements(const rm_model *model, size_t index);

/* One flat NHWC sample. Each concurrent invocation needs a distinct arena and output.
   No heap allocation in successful inference. The trace pointer is valid in the callback only. */
int rm_predict(const rm_model *model, const float *input, size_t inputs, float *output,
               size_t outputs, int8_t *arena, size_t arena_size, rm_trace_fn trace, void *user,
               char *error, size_t error_size);
#ifdef __cplusplus
}
#endif
#endif

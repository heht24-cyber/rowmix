/* Process the first little-endian float32 sample using the stable C entry points. */
#include "rowmix.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 3) { fputs("usage: rowmix-embed MODEL INPUT.f32\n",stderr); return 2; }
    char error[256] = {0};
    rm_model *model=rm_open_file(argv[1],error,sizeof(error));
    if (!model) { fprintf(stderr,"%s\n",error); return 2; }
    size_t inputs=rm_inputs(model), outputs=rm_outputs(model), bytes=rm_arena_bytes(model);
    float *x=(float*)malloc(inputs*sizeof(float)), *y=(float*)malloc(outputs*sizeof(float));
    int8_t *arena=(int8_t*)malloc(bytes);
    FILE *file=fopen(argv[2],"rb");
    int status=2;
    if (file && x && y && arena && fread(x,sizeof(float),inputs,file)==inputs &&
        rm_predict(model,x,inputs,y,outputs,arena,bytes,NULL,NULL,error,sizeof(error))==0) {
        size_t winner=0;
        for (size_t i=1;i<outputs;++i) if (y[i]>y[winner]) winner=i;
        printf("class=%zu arena=%zu backend=%s\n",winner,bytes,rm_backend(model));
        status=0;
    } else fprintf(stderr,"inference or input failed: %s\n",error);
    if (file) fclose(file);
    free(arena); free(y); free(x); rm_close(model); return status;
}

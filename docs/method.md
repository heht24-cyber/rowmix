# Calibration, allocation and the release gate

## Which rows receive eight bits?

Calibration runs the float graph on representative inputs. For every activation, try absolute-value clipping at the 99th, 99.9th and 100th percentiles and retain the scale with the smallest reconstruction MSE. Each output channel of every dense or grouped convolution gets two symmetric weight encodings. For a channel with weights w, test input activations X, and destination activation scale s, the local error proxy is mean((X(w−wq))²) / s². Convolution uses at most 8,192 deterministically spread input patches per group. The calibration code uses a Gram matrix where economical. The proxy is measured before the following nonlinearity.

For a row with K coefficients the upgrade costs K−ceil(K/2) bytes. Within a layer the price is constant, so the best choice of k upgrades is the prefix of the k largest positive reductions in local error. Dynamic programming combines layer prefixes, pruning higher-cost states with no higher gain. This optimizes the sum of local proxies among these choices under the exact serialized-file budget. Downstream quantization, correlated errors and class margins can change the classifier ranking of two candidates. The fixed int8 endpoint uses eight bits everywhere, even if a particular channel happened to fit four bits exactly.

The benchmark holds each layer's number of upgrades fixed and compares the chosen rows with a weight-MSE choice and eight random channel selections. These alternative allocations have exactly the same file size. Their held-out outcomes are included in results.json. Training labels and validation labels are unused by the compiler.

## Deployment selection

`rowmix-select` builds a fixed family of profiles using calibration data and deduplicates the resulting files by SHA-256 before looking at validation predictions. On the **separate validation set** it compares native top-1 predictions with the frozen float model. With n independent representative observations, k matching predictions, confidence 1−α and M distinct candidate files, the one-sided lower bound is zero for k=0 and otherwise BetaInverse(α/M; k,n−k+1). Bonferroni gives simultaneous coverage across candidates even if their mistakes overlap. Select the smallest file whose lower bound clears the requested agreement floor and whose device observations meet the supplied resource limits.

This estimates agreement with a reference model. Task accuracy still needs labels. The coverage statement assumes a fixed, independently constructed family and fresh independent validation observations from the target population. Reusing the validation cases to tune training, calibration or the candidate grid invalidates that interpretation. The two bundled studies illustrate the selection rule on fixed benchmark splits; they do not establish a guarantee on a new deployment. Per-device timing is empirical. The latency option limits the observed 95th percentile of the **mean per-sample time per pass**. It does not bound the tail latency of individual requests.

The CLI writes a JSON report for a failed quality gate, exits with status 3, and leaves any previously selected model intact. Confidence, number of candidates, agreement counts, failed limits, and model identities are included in the report.

[Exact binomial intervals, NIST](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbici.htm) · [ONNX Runtime quantization background](https://onnxruntime.ai/docs/how-to/quantization.html)

## Calibrating a memory-mapped dataset in batches

The compiler and ONNX importer accept `--calibration-batch N`. The selector accepts the same option for the calibration split. The Python API exposes `Calibration(g, x, batch_size=N, activation_samples=65536, max_patches=8192)`. The full-data path remains the default.

The streaming path knows the number and shape of samples from the mapped NumPy array. Before its first forward pass it chooses at most 65,536 scalar indices per node, using one pseudorandom index per equal-sized stratum (fixed seed). For layers below the limit it retains all values. Patch samples for row scoring use the same global evenly spaced indices as the full-data path, including per-group channel slicing. The indices refer to the whole dataset and stay fixed as batch boundaries move. All activations are checked for finiteness before sampling. NumPy/BLAS configurations may still produce small floating-point differences.

For batch size B, fixed-shape graph and fixed caps, retained calibration work arrays are bounded independently of the number of input examples. They include one batch of floating-point traces and temporary convolution matrices, the scalar samples per node, and up to max_patches feature vectors for each weighted layer and group. Wide kernels, deep graphs, the compiler's allocation search and filesystem page residency still affect process memory. The approximate activation scale can change which rows receive eight bits. The runtime arena and model format are shared with full calibration.

The Linux memory experiment starts a new process per configuration and records its lifetime peak resident set (KiB), elapsed compile time, file size and native predictions. Its larger input repeats the 259 Digits calibration observations four times to vary processing volume while controlling the data distribution. The same 360 held-out images evaluate all outputs. The repetitions add no independent observations. Reproduce with `python -m benchmarks.calibration_memory`.

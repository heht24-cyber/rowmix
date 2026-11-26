# RMX version 1

An RMX file is a self-contained, little-endian inference graph. All scalar floating-point fields are IEEE 754. The format implies dense int8 activation buffers, zero offset and symmetric range [-127, 127]. An input sample is float32 NHWC flattened in C order. A matrix input uses H=W=1. Each call returns flat float32 dequantized logits or activations. There is no batch dimension in the deployed graph.

## Ordered fields

| Region | Encoding |
|---|---|
| Envelope | Eight bytes `ROWMIX1\0`, then uint32 body length and uint32 IEEE CRC-32 of the body |
| Graph | uint32 arena bytes, node count, preprocessing channel count |
| Input transform | For each preprocessing channel, float64 mean and inverse standard deviation. Zero channels means identity. |
| Each node | Thirteen uint32 values, then one float64 activation scale |
| Each dense or convolution output channel | uint8 weight bit width (4 or 8), float64 weight scale, float64 real bias, followed immediately by its packed coefficients |

The node integers are op, fused-ReLU flag, parent A, parent B, output H, W, C, arena byte offset, kernel height, stride, symmetric padding, group count and kernel width. Missing parents are 0xffffffff. The first node is input and the last is output. Other nodes may only refer to earlier nodes. Ops 0–6 are input, dense, convolution, add, maxpool, ReLU and spatial global mean. Weights use output-major rows, then kernel y, x, and input channel within group. A dense row uses flattened NHWC order.

Four-bit values use two's complement, low nibble first. The compiler uses weights from −7 to 7 and zeroes the final unused nibble in an odd-length row. Eight-bit rows use signed bytes from −127 to 127. There is no alignment padding between records. Header, per-channel metadata, optional preprocessing and final odd nibbles all count toward a file budget. The checksum detects corruption; it does not authenticate a publisher.

## Arithmetic

Quantize a real value by dividing by the tensor scale, clipping to [−127,127] and rounding to nearest integer with half ties away from zero. Scale selection uses float64. An affine layer accumulates signed products in int32 and decodes each output with `acc * (input_scale * row_scale) + bias`. Apply a fused ReLU if enabled, then quantize to the destination scale. A dot has at most 100,000 terms, keeping even worst-case supported byte products inside int32. Entropy coding and additional weight unpacking buffers are unnecessary. Fused multiply-add contraction is disabled in the reference build to match Python's ordering.

Residual addition decodes both parents with their own scales. Padded max pooling excludes outside pixels. Global average sums integer values in int64 before scaling and division. The input preprocessing transform is `(raw - mean) * inverse_std` in float64.

## Memory and concurrency

The loader holds a copy of the encoded file and decoded row/node metadata. A caller supplies one byte arena per active inference plus separate float32 input/output buffers. The planned arena includes the live input and output activations. Nodes are born before their last-use parents are released. The allocator reuses gaps, and the loader rechecks all live overlaps. The live-volume lower bound is exact for the chosen execution order before placement constraints. The best-fit placement need not attain it.

The C API accepts an immutable loaded model shared by threads. Give every call its own nonoverlapping input, output and arena, keep the model alive for the duration of each call, and copy trace values in the callback if they are needed after it returns. The inference hot path makes no heap allocations. Standard library, loader metadata, file buffers and call stacks are additional to the arena.

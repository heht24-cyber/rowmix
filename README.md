# RowMix

RowMix 把训练好的卷积网络或多层感知机编译成可在 CPU 上独立运行的模型。给定浮点模型、代表性输入和**整个模型文件的字节上限**，编译器为每个输出通道选择 INT4 或 INT8 权重，输出一个 `.rmx` 文件。部署端只需要 C++17 运行时，也可以从 C 调用。

它适合研究一个具体问题：在存储和工作内存受限的 CPU 上，一个模型究竟能压到多小，还能维持多少预测一致率？项目提供训练代码、两份训练好的模型、独立 ONNX Runtime 对照、逐层整数结果以及没有通过验收的记录。

下载含提交历史的源码压缩包后，解压即可构建。若要继续开发，在解压目录执行 `git clone history/rowmix.bundle ../rowmix-dev` 可恢复主分支、版本标签和完整提交历史。

## 先运行随包模型

Linux 或带 GNU Make 的类 Unix 环境，在源码目录执行：

```sh
make -j2
./build/rowmix-run inspect examples/digits/mixed.rmx
./build/rowmix-run run examples/digits/mixed.rmx examples/digits/sample.f32 predictions.f32
cmp predictions.f32 examples/digits/expected.f32
```

示例输入包含八张手写数字图像。输出依次为每张图的十个 little-endian float32 分类分数。`cmp` 成功时没有输出。运行示例无需 Python 或网络。

Windows 可使用 Visual Studio 的 C++ 工具链和 CMake，在项目目录执行：

```powershell
cmake -S . -B cmake-build
cmake --build cmake-build --config Release
ctest --test-dir cmake-build -C Release --output-on-failure
.\cmake-build\Release\rowmix-run.exe inspect examples\digits\mixed.rmx
```

在 C 或 C++ 程序里嵌入时，先加载模型，按 `rm_arena_bytes` 分配一次激活缓冲区，再对每个样本调用 `rm_predict`。多个线程可共享只读模型，各自持有缓冲区。完整代码在 [examples/embed.c](examples/embed.c)，接口在 [native/rowmix.h](native/rowmix.h)。

## 编译自己的模型

编译器需要 Python 3.10 以上。普通安装只需要 NumPy；下例额外安装 ONNX 导入器和模型选择器依赖。以下为 shell 命令。

```sh
python -m pip install ".[onnx,tuning]"
rowmix examples/digits/float.npz examples/digits/calibration.npy result.rmx --bytes 8726
./build/rowmix-run inspect result.rmx
```

编译自有 ONNX 时，校准输入遵循原模型的 NCHW 或 NC 布局。下例将附带的 Digits 扁平数组恢复为 NCHW，再分批校准，每批八条输入：

```sh
python -c "import numpy as np; x=np.load('examples/digits/calibration.npy'); np.save('digits-nchw.npy',x.reshape(-1,1,8,8))"
rowmix-onnx examples/digits/model.onnx digits-nchw.npy imported.rmx --bytes 8726 --calibration-batch 8
./build/rowmix-run run imported.rmx examples/digits/sample.f32 imported-scores.f32
```

ONNX 前端支持单输入、单输出、固定非批次维度的 float32 图。支持全连接、二维卷积（含分组、逐通道和矩形卷积核）、残差相加、ReLU、最大池化、全局平均池化，以及受限的 Flatten、Reshape 和归一化折叠。卷积要求各空间方向步幅相同、对称填充、dilation 为 1。遇到不支持的算子会报告节点。运行时的输入布局为 NHWC，矩阵输入为 NC。当前覆盖范围是 CNN 和 MLP。大型语言模型和动态序列模型超出本版的验证范围。

## 预算和验收如何工作

1. 编译器用校准激活衡量每行权重从四位升到八位能够减少多少局部输出误差，同层按收益排序，再用动态规划在总文件预算内分配升级名额。目标函数是各层局部误差之和。任务准确率另用留出样本检验。
2. 运行时直接读取打包权重，在支持的 x86 CPU 上选择 AVX2，也提供标量实现。点积用整数累加，偏置、缩放和重新量化使用浮点。按激活生命周期规划的缓冲区在成功推理期间可重复使用。
3. `rowmix-select` 在观察验证结果前生成候选，再用本机运行时测量模型文件、激活内存、推理耗时及与浮点模型的 top-1 一致率。以单侧精确二项置信下界校验一致率，并校正候选数量。可行时输出满足条件的最小文件；无候选通过时退出码为 3，写出原因并保留原有模型。置信解释要求验证样本独立且能代表目标分布。

如有独立的 NHWC/NC 验证输入 `heldout.npy`，可执行：

```sh
rowmix-select examples/digits/float.npz examples/digits/calibration.npy heldout.npy \
  --runner ./build/rowmix-run --min-agreement 0.95 --confidence 0.95 \
  --max-bytes 9000 --max-arena 2400 --output selected.rmx --report selection.json
```

## 实测结果

训练、校准和测试使用不同样本。下表使用 INT4 与 INT8 两端文件大小中点作为预算。激活区指 C API 调用方需提供的工作缓冲区；进程还需要加载模型、代码及输入输出。

| 模型 | 浮点 ONNX 文件 | RowMix 文件 | 激活区 | 浮点预测正确 | RowMix 预测正确 | 与浮点类别相同 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Digits 残差 CNN | 37,885 B | 8,710 B | 2,368 B | 356/360 | 356/360 | 358/360 |
| Wine MLP | 4,649 B | 2,081 B | 61 B | 35/36 | 35/36 | 36/36 |

- **压缩与速度。** Digits 混合模型含 31 行 INT4 和 47 行 INT8。在 AMD EPYC 9V74 单线程 AVX2 上，预热后遍历测试集 21 次，各次每样本平均耗时的中位数为 79.13 µs。全 INT8 模型占 11,014 B，同一实验为 74.61 µs。
- **分配策略。** 固定各层升级行数和文件字节数，按纯权重误差选行得到 357/360，按校准激活误差选行得到 356/360，八次随机选行为 354–356/360。这个样本集没有显示激活误差策略在任务准确率上领先。
- **验收也能失败。** 在五个候选中，Digits 的 7,576 B 模型与浮点结果一致 359/360，校正后下界为 98.17%。Wine 的五个候选均为 36/36，下界只有 87.99%，未达到设定的 95% 要求。该规则衡量模型间一致率，任务准确率由标签单独衡量。
- **校准内存。** 在 Linux 上将固定的 259 条校准输入重复四遍；每种配置使用独立新进程。处理 1,036 条输入时，全批校准与八条分批校准的峰值 RSS 分别为 136.3 和 51.2 MiB，模型分别为 8,710 和 8,722 B，两者在 360 张留出图像上均预测正确 356 张。分批校准对每层至多采样 65,536 个激活值，适用于校准内存有限的机器。重复输入用于测工作量和内存，独立校准图像仍为 259 张。

查看 [逐预算测试、延迟与消融](benchmarks/results.json)、[模型选择记录](benchmarks/selection.json)、[ONNX 差分结果](benchmarks/interchange.json) 和 [校准内存原始数据](benchmarks/calibration-memory.json)。

## 从头验证

参考环境为 Linux x86-64、Python 3.12、C++17。安装固定依赖后可执行：

```sh
python -m pip install -r requirements-validation.txt
export OPENBLAS_NUM_THREADS=1
make test
python -m benchmarks.run --reuse --passes 21
python -m benchmarks.interchange
python -m benchmarks.selection
python -m benchmarks.calibration_memory
```

去掉 `--reuse` 会重新训练两个模型。测试覆盖预算分配与小规模穷举的一致性、内存生命周期、并发 C ABI、分批校准和失败时的输出完整性，还将原生逐层结果同独立 NumPy 实现比较。ONNX 对照读取已安装的 ONNX 1.18.0 所附 14 组上游用例，并在两个完整模型上与 ONNX Runtime 对照浮点转换误差。机器和依赖版本会影响耗时和浮点末位。

## 后续研究

下一阶段以可复现的公开预训练视觉模型为验收对象。先记录导入时遇到的第一个不支持的算子或布局，再按实际图补齐；对完整验证集分别报告精度、与浮点模型的分歧、文件大小、峰值校准内存及原生延迟。随后在 ARM64 实机上把同一份模型分别交给标量与 SIMD 后端，检查每一层的整数输出。以上是后续里程碑，本版证据限于随附结果与运行环境。

[算法及统计假设](docs/method.md) · [二进制格式](docs/format.md) · [数据来源](NOTICE.md)

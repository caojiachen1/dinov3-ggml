# dinov3-ggml 性能提升计划

> 状态：规划中，尚未动任何代码。
> 基线仓库：本仓库 HEAD `153fba4`（与 differ-rs Cargo.lock 锁定版本一致）。
> 目标硬件：RTX 5080（sm_120，16GB），Windows + MSVC + CUDA Toolkit。

---

## 1. 目标

| 指标 | 当前基线（RTX 5080） | 目标 |
|---|---|---|
| 单图延迟（含解码+resize） | 28.1 ms avg / 30.1 ms p95 | ≤ 15 ms |
| 批量吞吐（batch_extract） | 46.3 img/s（21.6 ms/img） | ≥ 100 img/s |
| 模型加载 | 303 ms | ≤ 300 ms（不回退） |
| 精度 | 与 HF 参考实现 cosine ≈ 0.999999 | 精确等价类改动 bitwise/1e-6 级不回退；近似类改动 cosine ≥ 0.999 且出报告 |

目标数值在 M0（瓶颈定位）完成后按实测拆分修订，此处是方向性目标而非承诺。

## 2. 基线与关键判断

### 2.1 实测基线（differ-rs `bench_perf`，RTX 5080）

- GGML CUDA：单图 extract avg **28.10 ms**、min 26.27、p95 30.12；顺序 8 图 10.0 img/s；batch 200 图 **46.3 img/s**
- 对照组 ONNX DirectML：单图 54.1 ms、batch 23.0 img/s（GGML 已快一倍）
- CUDA graph 已启用（日志可见 `CUDA graph warmup complete`）
- 已知异常：启动时出现 `ggml_gallocr_needs_realloc: graph has different number of nodes` + 自动 realloc

### 2.2 Roofline 估算（为什么值得做）

ViT-S/16，12 层，H=384，I=1536，token 数 1029（1 CLS + 4 reg + 32×32 patch）：

- GEMM FLOPs ≈ 2 × 21.5M 参数 × 1029 token ≈ **44 GFLOPs**
- attention FLOPs ≈ 4 × 1029² × 64 × 6 heads × 12 层 ≈ **20 GFLOPs**
- 合计 ≈ **64 GFLOPs/图**。5080 级 F16 张量核心理论 >100 TFLOPS，按小 GEMM 有效算力 10–30% 保守估计，**GPU 计算下限约 2–6 ms/图**

实际 21.6 ms/img，即 **4–10 倍的优化空间不在 GEMM 数值计算本身**，大概率分布在：

1. **CPU 预处理**（JPEG 解码 + `image` crate 标量 Lanczos3 resize，单线程各需十几到几十 ms；batch 路径靠 rayon 并行摊薄到 21.6 ms/img，说明预处理很可能就是当前吞吐的墙）
2. **小算子开销**：每层约 25–30 个 kernel（norm、bias add、permute/cont、RoPE 的 roll+mul+mul+add×2、cast×2、fattn、GELU、残差 add×4…）× 12 层 ≈ 300+ 个 kernel。每个 elementwise kernel 读写 [384×1029]≈1.6 MB，内存时间仅 ~3 µs，但 launch + 收尾在 CUDA graph 下仍有 ~5–10 µs，合计 **2–4 ms 纯开销**；RoPE 一项就占每层 8 个 kernel
3. **重复的 layout/dtype 转换**：`cont(permute(x))` + `cast(F16)` 是两次全量拷贝；`ggml_cpy` 一个 kernel 就能完成 permute+cast+cont

结论：**优先级 = M0 先用数据确认拆分 → M2 预处理（大概率最大头）→ M3 算子融合/裁剪 → M1 快赢项穿插 → M4 量化收益存疑放最后**。

交叉验证一条矛盾信息：实测批量 46.3 img/s（21.6 ms/img）是在 `max_batch=1`、GPU 串行逐张的前提下拿到的。若"CPU 预处理并行容量约 100–350 img/s"的估算成立，则当前的墙必然在 GPU 侧每张 ~21.6 ms——这与"GPU 前向只要 2–4 ms"的估算直接矛盾。两个估算不能同时为真（另一种可能：预处理远比估算慢）。这正是 M0 要裁决的问题，**M2 与 M3 的先后由 M0 数据定**。

### 2.3 已有优化盘点（不要重复做）

- 权重存 F16（省一半带宽 + tensor core）；CUDA graph 开启（`GGML_CUDA_GRAPHS=ON`）
- LayerNorm affine + LayerScale 在加载期折叠进 GEMM 权重/bias（省每层 6 个 elementwise pass）
- 计算图预构建复用（单图 + 可选 batch 两套，输入尺寸固定）
- RoPE cos/sin 预计算成表，rotate-half 符号折进 sin 表；CLS/register 行为恒等
- flash attention（`ggml_flash_attn_ext`），K/V 转 F16
- batch 路径把 B 折进 token 维，全层 GEMM 当一个 2D 大算子跑
- `extract_batch` 生产者-消费者流水线：rayon 预处理领先 GPU 最多 2 个 chunk

---

## 3. 阶段计划

### M0：环境准备与瓶颈定位（先做，半天–1天）

**环境（只做准备，不改代码）：**

1. `test/` 放入测试图（从 differ-rs/test 拷 1.jpg–5.jpg，本仓库 .gitignore 已忽略 `/test`）
2. `models/dinov3_vits16.bin` 就位（从 differ-rs/models 拷贝或 ModelScope 下载）
3. 构建验证：`cargo build --release --features cuda`；确认 build.rs 的 `detect_cuda_arch()` 返回 `120`（sm_120 原生 SASS，而非 PTX JIT）
4. differ-rs 联调通路：在 differ-rs 根 Cargo.toml 加 `[patch."https://github.com/caojiachen1/dinov3-ggml"] dinov3-ggml = { path = "../dinov3-ggml" }`，即可用本仓库跑 differ-rs 的 bench_perf / compare_demo

**测量（本阶段交付物 = 时间拆分表）：**

| 测什么 | 怎么测 |
|---|---|
| 纯 GPU 推理（不含预处理） | 复用 differ-rs `bench_ggml.rs` 的思路：`preprocess_image` 一次，循环 `infer` 计时（该 example 已存在，直接跑） |
| JPEG 解码 / resize / 归一化各自耗时 | 在临时 example（或 demo 加开关）里分三段 `Instant` 计时；只加测量代码不算动刀，但先在这里只记录数字 |
| GPU 上传/下载耗时 | 3.2 MB 上传 + 1.58 MB 下载，理论 <1 ms，实测确认可忽略 |
| 每类算子 GPU 耗时 | `ggml_backend_synchronize` + CUDA 事件，或 nsys profile `ggml_backend_graph_compute`，定位 kernel 直方图 |
| batch 吞吐随 max_batch 变化 | `GGML_VIT_MAX_BATCH=1/2/4/8/16` 各跑一遍 batch 200 |

**判据（决定后续投入方向）：**
- 若纯 GPU 推理 ≤ 8 ms → 主攻 M2 预处理 + M5 流水线；M3 只做 RoPE/cpy 两个高性价比融合
- 若纯 GPU 推理 > 12 ms → M3 提到与 M2 并行，重点查 fattn 和 mul_mat kernel 效率
- 若 max_batch ≥ 4 显著提升吞吐 → M5 的自动 batch 策略提优先级

### M1：快赢项（低风险，随做随验，合计半天–1天）

| # | 改动 | 预期 | 风险 |
|---|---|---|---|
| M1.1 | 排查 `ggml_gallocr_needs_realloc` 告警来源（为何单图图分配后还有一次不同节点数的分配；确认是否只发生一次、是否引入了额外全图分配） | 消除启动期一次多余 realloc；排除隐患 | 低 |
| M1.2 | GGML submodule `9be3133` → 最新 master，跑基线对比。Blackwell(sm_120) 较新，上游可能有 CUDA kernel 修复/改进；注意 `ggml_roll`、`ggml_backend_cpu_set_n_threads` 等较新 API 的兼容 | 未知，可能白拿几个百分点 | 中：API 变动需小改 C 侧；必须全量跑精度门 |
| M1.3 | 确认 nvcc 产物含 sm_120 原生 cubin（`cuobjdump`），不是 PTX JIT 运行 | 消除潜在 JIT/兼容 kernel 慢路径 | 低 |
| M1.4 | batch 吞吐模式下把 `max_batch` 从"仅环境变量"提升为 `VitConfig` 正式字段（现在 lib.rs 只读 env，config 里的字段没接上）+ 吞吐模式默认值建议 | 接口清晰化，配合 M5 | 低 |
| M1.5 | CPU-only 路径：`get_num_cores` 用逻辑核数，混合架构大小核可试物理核数对比 | CPU 路径小幅提升（次要目标） | 低 |

### M2：预处理优化（预计最大收益，1–2 天）

现状：`image` crate 解码 → `resize_exact(Lanczos3)`（标量实现，~4000×3000 源图单线程可达 40–80 ms）→ `to_rgb8` → HWC→NCHW 归一化（又一趟全像素遍历）。

| # | 改动 | 预期 | 风险/约束 |
|---|---|---|---|
| M2.1 | resize 换 `fast_image_resize`（SIMD）。两个候选滤波器：(a) Lanczos3 保持现语义；(b) CatmullRom（bicubic）——HF 官方 DINOv3 预处理就是 bicubic，既更快又更贴参考实现，倾向 (b) | resize 40–80 ms → 5–10 ms，**单图延迟最大单项** | 已核实预处理存在**两份副本**：differ-rs 的 onnx/candle 用其自带 `src/dinov3/preprocessing.rs`，ggml 后端走本 crate 的 `preprocessing.rs`。换滤波器必须两处同步，否则跨后端 compare 失效；与 `image` 实现非逐位一致 → 走 Gate B |
| M2.2 | 融合后处理：从 resize 后的 RGB8 缓冲一趟循环直接产出归一化 CHW f32（现在 to_rgb8 + 双层循环已是单趟，但换成 fir 后重写这一段并按通道Splitting并行） | 每图省 1–2 ms + 少一次中间分配 | 低 |
| M2.3 | 解码提速：JPEG 已走 zune-jpeg（image 0.25 默认，本身不慢）；大图（≥8MP）可再上 turbojpeg 的 1/2、1/4 **DCT 缩放解码**（解码期直接降采样，像素量减 4–16 倍，再交给 resize 收尾到目标尺寸，对"解码+resize"合计收益大）；PNG 走 `png` crate 较慢，若实际工作负载含大量 PNG 再引入 zune-png/libpng-turbo | 视负载与源图尺寸 | 中：新依赖 + 半解码后需重验 resize 链路精度（Gate B） |
| M2.4 | 上传减半：输入张量改 F16（GGML 图输入侧支持 F32→F16 上传转换，或 Rust 侧直接产出 F16 缓冲） | 上传 3.2→1.6 MB，省 <0.5 ms | 需精度评估（输入动态范围小，F16 足够）；收益小，排后 |
| M2.5 | 单图路径也走流水线：decode 与（上一次请求的）GPU 推理重叠——仅在服务化/连续请求场景有意义，视 differ-rs 调用模式决定 | 连续请求场景隐藏预处理 | 依赖 M0 结论 |

排序依据：M2.1 单项收益最大且改动局部（`preprocessing.rs` 一个函数），先做。

### M3：GPU 算子层（1–2 天，按 M0 结论取舍）

| # | 改动 | 预期 | 风险/约束 |
|---|---|---|---|
| M3.1 | **RoPE 输入改 F16**：Q/K 在 permute 后立即 cast F16，roll+mul+add 全链路 F16，cos/sin 表也转 F16。省掉 K/V 各自的独立 cast kernel（每层 2 个），RoPE 的 8 个/层 elementwise 带宽减半 | 每层省 2 kernel + RoPE 带宽减半，合计估 1–2 ms/图 | 数值近似 → Gate B；fattn 接受 F16 Q（llama.cpp 同款做法） |
| M3.2 | **`ggml_cpy` 替换 `cont(permute)+cast` 两连击**：V 的 `cast(cont(permute(Vv)))` 与 K 的 cont 后 cast，均可并为一次 `ggml_cpy`（一个 kernel 同时做 layout 转换 + dtype 转换） | 每层省 2–3 个全量拷贝 kernel | 精确等价类（cpy 就是逐元素转换）→ Gate A |
| M3.3 | RoPE 自定义算子（GGML custom op）：把 roll+mul+mul+add 4 kernel 融成 1 个 | 每层 8 kernel → 2，估省 1–2 ms/图 | 工作量最大的一项；CPU 回退实现要写；放 M3.1/M3.2 验证后评估是否仍值得 |
| M3.4 | bias-add 折叠进 GEMM（tokens 增广常数 1 列 + W 拼接 b 列）：注意 norm 是按 token 行算的，增广列必须在 norm 之后 concat，等于"add 换 concat"，带宽同级 → **先在纸上推演，大概率不做**；若做需 benchmark 佐证 | 未必有收益 | 中 |
| M3.5 | patch embed conv2d 确认走 im2col+GEMM 高效路径（PS=16 大 kernel，im2col 后 768×1024 的 GEMM，应无问题），M0 的 kernel 直方图里若占比异常再处理 | — | 低 |

### M4：量化权重（存疑，最后做，1 天评估）

- F16 → q8_0 / q4_0：本模型 N=1029 时 GEMM 偏计算受限，MMQ int8 未必快过 F16 tensor core；q4_0 省一半权重带宽但精度损失需报告
- 对外部"INT8 tensor core 吞吐 2× FP16 → 预期 1.5–2×"意见的评估：仅在 GGML CUDA 命中 int8 mma kernel（较新版本才有）且 M 维大（batch ≥ 8）时可能成立；单图/小 batch 下大概率无收益甚至回退。维持"先 M0 证据后投入"
- 前置条件：M0 显示 mul_mat 确实是 GPU 热点才投入；做成 opt-in feature + 精度报告（本仓库卖点是数值对齐 HF，量化只能作为可选项）
- 需要改 `convert_weights.py` + `ggml_vit.c` 读入路径 + load 期折叠逻辑适配

### M5：吞吐/流水线（半天–1 天，按 M0/M1 数据）

| # | 改动 | 预期 |
|---|---|---|
| M5.1 | 吞吐模式自动 max_batch：M1.5 扫描结果落成默认值（如吞吐默认 8、延迟默认 1），暴露到 `FeatureExtractor::with_batch()` | batch 46 → 视 M1 扫描，可能 80–150 img/s |
| M5.2 | 双缓冲上传：CUDA stream 上 overlap 第 i+1 批的输入上传与第 i 批计算（当前 `tensor_set` 是同步的） | 上传隐藏，估 <1 ms/批；收益小，靠后 |
| M5.3 | 流水线深度与 chunk 策略：`chunk_size = max(max_batch, 16)` 的 16 是拍脑袋值，按 M0 的预处理耗时调（chunk 应保证 rayon 打满所有核且不超过背压 2 chunk 的内存） | 消除 GPU 空泡 |

### M6：应用层特征池化（differ-rs 侧改动，不在本 crate，优先级独立于 M0–M5）

现状（已核实代码）：`differ-rs/src-tauri/src/services/similarity_service.rs` 全程使用全量 token 特征 [1029, 384] = 395,136 维：

- `l2_normalize` 整向量归一化；`search_similar_in_folder` 一对多全维 cosine；`compare_folders` 更是 source×target 全叉积
- `cache_service` 把全量特征按 ~1.5 MB/图 存进 SQLite

代价量级：1 万张图 ≈ 15 GB 缓存；两两比对 10⁸ 对 × 393K 维 ≈ 4×10¹³ FLOPs 纯 CPU——**比对成本比推理本身高几个数量级**，这才是应用吞吐的真正大头。

方案：CLS token（DINOv3 官方全局相似度推荐）或 patch mean-pool → 384 维，存储与比对成本缩 ~1029 倍；全量特征保留为可选输出（仅 copy-move / patch 级检测需要）。落点：

1. differ-rs：`similarity_service` + `cache_service`（缓存 schema 需版本化，旧缓存失效升级）
2. 本 crate（可选）：加 pool 输出选项，GPU 侧池化后每图只下载 1.5 KB（省 IO，不省计算）
3. 精度验证：differ-rs 测试集上对比"全量特征 cosine 排序"与"CLS / mean-pool cosine 排序"的 top-k 一致性，再决定 CLS 还是 mean-pool

### M7：输入分辨率档位（产品级选项，M2/M3 落定后评估）

查重场景不需要 518²：256²（16×16 网格 → 261 tokens）约省 3.6×，224²（14×14 → 201 tokens）约省 4.8×，attention 部分按 token² 下降更快。RoPE 位置编码对分辨率天然友好（patch 坐标归一化到 [-1,1]，无需 pos-embed 插值），架构上干净，只需按分辨率各建一套预建图（RoPE 表 + grid 按输入尺寸预计算，权重共享）。做成 fast(256) / high(518) 两档；前置条件是用 differ-rs 测试集验证近重复检测在低分辨率下的排序一致性。与 M6 叠加后，单图端到端有望进 10 ms 档。

---

## 4. 精度回归门（每个改动都过，不过就回退）

参照物：当前 HEAD `153fba4` 在固定测试集（differ-rs/test 5 张图 + 本仓库 test/）上的输出特征与相似度矩阵，先跑一次存档为 golden。

- **Gate A（数值精确等价类：cpy 替换、图结构等价重构、批处理拆分）**：与 golden 逐位一致或 max abs diff < 1e-6
- **Gate B（近似类：F16 RoPE 表、fast_image_resize、量化）**：与 golden cosine ≥ 0.999（逐特征向量），报告 max diff；differ-rs `compare_demo` 的跨后端 max |sim diff| 不劣化超过现有阈值
- 现有单测（`cargo test`）全绿；`test_preprocess_output_shape` 等布局测试不破坏
- 每项改动单独提交，单独跑 `bench_perf`，结果记入本文件附录（追加表格），保证可归因、可回退

## 5. 基准方法（每项改动统一口径）

```bash
# 本仓库侧（改动的第一落点）
cargo run --release --features cuda --example demo          # 冒烟 + 精度肉眼检查
# differ-rs 侧（联调 + 对照）
cd D:\Codebase\differ-rs
bench_perf.bat 8          # 快速：8 图顺序 + batch
bench_perf.bat 200        # 完整：单图延迟 30 iters + batch 吞吐
target\release\examples\compare_demo.exe   # 跨后端精度
```

记录格式：日期 / commit / 改动项 / 单图 avg·min·p95 / batch img/s / 精度门结果。

## 6. 风险与注意事项

1. **本仓库 `.gitignore` 忽略 `*.md`（除 README）**：本计划文件默认不入库；要提交需 `git add -f PERF_PLAN.md` 或改 ignore 规则
2. **submodule 升级（M1.2）是最大的不确定性来源**：单独分支做，先跑 Gate A 全量
3. **跨后端可比性**：改预处理滤波器/精度会同时影响 differ-rs 里 ONNX/Candle 后端的对比基线——differ-rs 的 Rust 预处理如与本仓库共享实现需同步；若各自实现则只动本仓库会破坏 compare 口径，M2.1 实施前先确认
4. **CUDA Toolkit 版本**：sm_120 需要较新 nvcc（CUDA 12.8+），M1.3 一并确认
5. batch 图的 stale-slot 策略（末批不足 max_batch 时补旧图）是正确取舍，保持不动
6. 所有 `getenv`（GGML_VIT_MAX_BATCH / GGML_VIT_FORCE_CPU）行为保持向后兼容

## 7. 建议执行顺序（总结）

```
M0 定位（拆时间） ──┬─ 预处理是大头 → M2.1 → M2.2 → M5.1/M5.3
                    ├─ GPU 算子是大头 → M3.1 → M3.2 →（视结果）M3.3
                    └─ 都不大/到处都是 → M1.2 submodule + M1.3 arch 先排查再回来
穿插：M1.1 告警、M1.4 batch 配置化（半天内）
并行：M6 应用层池化（differ-rs 侧，不依赖 M0，用户体感收益最大）
暂缓：M3.4（先纸推）、M4 量化（等 M0 证据）、M2.3/M2.4/M5.2（小收益垫底）、M7 分辨率档位（M2/M3 落定后评估）
每步：单独 commit + Gate A/B + bench 记录进附录
```

---

## 附录 A：基准记录（随改动追加）

基准环境：RTX 5080（16GB）+ 16 逻辑核 CPU，ViT-S/16，`examples/bench`（n=图片数，progressive=test/（690px 网图），baseline=test_baseline/（同图重编码 baseline））。端到端含解码+resize+归一化+GPU 推理（流水线）。

### 2026-09-30 实施记录（M0/M2.1/M5/M7 完成，目标 ≥1000 img/s 达成）

| 配置 | preproc (img/s) | GPU only (img/s) | 端到端 (img/s) | 备注 |
|---|---|---|---|---|
| 518² progressive B1（原始基线） | 437 | 330 | **277** | 最初状态 |
| 518² progressive B4 | 516 | 398 | **319** | 518 档最优 batch=4 |
| 256² progressive B16（原始预处理） | 670 | 1400 | 508 | fast 档 GPU 上限验证 |
| 256² progressive B64（+fast_image_resize） | ~900 | ~1600 | **741** | progressive 解码 8.4ms/图为墙 |
| 256² baseline B32（+fast_image_resize） | 2280 | 1626 | **1282** | |
| **256² baseline B32, n=1024（最终）** | — | — | **1429–1465** | 3 次稳定复现 ✓ 目标达成 |
| 518² baseline B4（高精度档最终） | 1630 | 368 | **351** | GPU-bound，贴近预估上限 |

各阶段单线程耗时（256²，per img）：decode progressive 8.4ms / baseline 2.9ms；resize+normalize 原实现 4.9ms → fast_image_resize 后 **1.0ms**（-79%）。

关键发现与决策：

1. **differ-rs 早前测的 46 img/s 不是 GPU 墙**：GPU-only 在 518² 就有 330–400 img/s（2.5–3ms/图），当时瓶颈是大图的 CPU 预处理与流水线
2. **fast_image_resize（M2.1）**：resize+norm -79%，Gate B 通过（相似度第 4 位小数级偏移 <0.0005）
3. **turbojpeg 实测回退**：libjpeg-turbo 在 progressive JPEG 上 10.7ms vs zune-jpeg 8.4ms，**更慢**，已移除（NASM 已装、现已不需要，`scoop uninstall nasm` 可清理）
4. **batch 拉满（M5）**：256² 最优 max_batch=32（环境变量 `GGML_VIT_MAX_BATCH=32`）；518² 最优 4
5. **fast 档（M7）**：256² 是达成 1000+ 的必要条件（518² GPU 上限 ~400）
6. **progressive vs baseline JPEG 是解码器之墙**：zune-jpeg 对 progressive 8.4ms/图（baseline 2.9ms）。网图为主的库 ~740 img/s；相机/手机原图（baseline）1400+
7. **流水线重叠效率 ~80%**：端到端落后 preproc 能力约 20%，后续可在 extract_batch 消费侧（FFI 拷贝/分配）继续优化

⚠️ 基准操作坑：`cargo test` / 不带 `--features cuda` 的 `cargo build` 会用无 CUDA 配置**覆盖** `target/release/examples/*.exe`（症状：输出无 `ggml_cuda_init` 横幅、[load] 从 ~240ms 掉到 ~60ms、速度掉到 ~20 img/s）。跑基准前确认输出含 CUDA 横幅，或始终带 `--features cuda`。

### 2026-10-01 真实负载校准 + Strata 策略吸收（crate 88a881f）

**用户实测 400 img/s 的解释（已复现）**：此前基准用 2000 张同一 690px 小图拷贝（页缓存热、解码占比≈0），测的是流水线上限。真实图库（唯一大图）完全卡在 JPEG 熵解码 CPU 时间 ÷ 核数。构造 200 张唯一 12MP（4000×3000）图集实测：**baseline 141 img/s、progressive 58 img/s**——用户 400 对应 2–6MP 图，数字完全对得上。

**Strata 策略逐项吸收结果**（prefill 8K 大块 / PCIe 与计算重叠 / helper 拷贝线程 / MMQ 量化 / 多卡流水线）：

| 策略 | 结论 |
|---|---|
| 大块 prefill | 已吸收：应用分块 8→512、GPU batch 扫描定案 B=32（48/64 更差） |
| PCIe 传输与计算重叠 | 已验证非瓶颈（CLS 直出持平）；读预取 depth-2 已在 |
| helper 拷贝线程 | 已在（后台缓存写 + 预取线程） |
| MMQ 量化 GEMM | 不适用（prefill 型计算受限负载，此前 M4 已论证） |
| 多实例/多流 | **负结果**：单卡双 GGML 实例被驱动串行化，589 vs 1611 img/s（examples/multi_inst.rs 存档） |

**224px ultra 档（GGML_VIT_TIER=ultra，opt-in）**：GPU-only 2108 img/s（256 档 1643、192 档 2876）；应用层小图 baseline 集 avg 1543 / 峰值 1750（fast 档 1170/1314）。排序结构保持（三对 sim 与 256 档同序）。默认仍 fast 档。

**2000 img/s 可达性结论**：仅对 ≤1MP 且页缓存热的图集，e2e 峰值 ~1750 逼近 GPU 上限 2108，更大文件夹可持续 ~1800；12MP 真实照片卡死在 JPEG 熵解码（~85ms/图 CPU ÷ 8 核 ≈ 180），唯一破局是 nvJPEG GPU 解码（消费卡无硬件 JPG 引擎，hybrid 模式预期 2–4×，独立立项）。

补充：turbojpeg fast chroma upsample 已开（1/8 缩放下为 no-op，1/2 有小收益）；oracle 重跑 PASS。

### 2026-10-01 桌面端集成冲刺（crate 473e240/11ee7f1/0f15ee6 + differ-rs 应用层）

crate 侧：DCT 缩放解码（fast 档默认，Gate B 校准：sim 偏移 ≤0.001、特征余弦 ≥0.9990）；加载期 CUDA graph 预热；`ggml_vit_infer_batch_cls` / `extract_batch_cls`（CLS 直出，显存→内存搬运量 -99.9%，位级校验与全量输出前 H 维一致）。scan_limit 实测负收益（progressive 熵解码不可跳过），默认 0。

differ-rs 侧（app_bench = 应用真实路径，1000–2000 张文件夹）：

| 场景 | 冷提取 | 缓存命中 |
|---|---|---|
| baseline JPEG 2000 张 | avg 1170 / 稳态 ~1300 img/s | ~16000 img/s |
| progressive JPEG 500 张 | ~584 img/s | ~15400 img/s |
| 混合 1000 张 | ~777 img/s | ~16000 img/s |
| bench_perf（200 progressive，缩放解码后） | 701 img/s（此前 649） | — |
| high 档 518² + CLS | 289 img/s | — |

对 1400 的结论：crate 层（图集预载入内存）1429–1465 仍成立；应用层再加真实磁盘读 + 目录扫描 + 2000 次 stat + 分块边界气泡后，稳态天花板 ~1300（已验证与显存下载无关——CLS 直出持平）。要应用层破 1400 剩下的路：NVMe 真实图库 + 更大文件夹摊薄固定开销，或把 Phase-1 缓存探测流水化（估计 +3–5%），不在本轮范围。

应用层其他落地：进度事件带 img/s + elapsed（StatusBar 实时显示）；SQLite WAL + 后台写线程（比对/缓存写不再阻塞提取）；compare_demo 强制 GGML_VIT_TIER=high 修数值口径（三后端 sim spread ≤0.0002）。

### 2026-10-01 differ-rs 集成（VitConfig.max_batch=845d7cf，预处理同步）

differ-rs `bench_perf`（test/ 为 5 张 690px progressive JPEG，RTX 5080）：

| 配置 | 单图延迟 | 批量吞吐 | vs 集成前 |
|---|---|---|---|
| ggml fast 档（新默认：256² + B=32 + SIMD 预处理） | 8.91 ms | **649 img/s** | 46.3 → **14×** |
| ggml high 档（GGML_VIT_TIER=high：518² + B=4） | 12.16 ms | 312.7 img/s | 28.1ms → 2.3× |
| onnx（518²，differ-rs 预处理同步 SIMD 后） | 42.2 ms | 29.4 img/s | 23 → +28% |

配套改动：differ-rs 预处理副本同步 fir（跨后端口径一致）、缓存按特征维度校验（换档自动失效重提取）、EXTRACT_CHUNK 8→64（喂满 B=32）。649 vs 理论 ~741 的差值 = progressive 解码墙（同 crate 侧结论）；baseline JPEG 图库走同路径 ~1300+ img/s。

### 2026-10-01 GPU 深度优化：F16 激活流 + vision RoPE + cublasLt epilogue（本轮）

**方法**：新增 per-op GPU 剖析（ggml 补丁 `GGML_CUDA_PER_OP=1`，CUDA event 逐节点计时）+ `examples/gpuprof.rs`（纯 graph 重放计时，预热 10 次排除 graph instantiate/re-capture 一次性开销）+ `examples/dumpfeat.rs`（Gate B 特征转储）。基线以纯 graph 重放口径重测（此前 bench [infer] 混入全量输出下载与分块气泡，不可比）。

**基线剖析（256² B32，每 eval 14.78 ms）**：MUL_MAT 6.2（含每次 GEMM 前后的隐藏 dtype 转换：F32→F16 src1 暂存 + F16→F32 输出回转，合计每层 ~460 MB 纯转换流量）+ elementwise/拷贝 8.7（59%，其中 RoPE 的 roll+mul+mul+add×2 链 4.9）+ fattn 1.5。

**改动**（ggml-src submodule 782e3662+b140148 + csrc 重写）：

1. **F16 激活流（CUDA 路径）**：`ggml_mul_mat_out()` 显式结果 dtype；cuBLAS 直写 F16（F32 累加），双向隐藏转换全消。norm 补 F16 in/out kernel；bias/cls/reg 前缀/patch 输出全 F16。CPU 回退保持全 F32 图（CPU mul_mat 深度绑定 F32 dst）
2. **vision RoPE 替换 roll 链**：发现本版 ggml 自带 `GGML_ROPE_TYPE_VISION`（Gemma3n 视觉模式）与 DINOv3 RoPE 数学完全同构（sections [D/4,D/4]、theta=pos·base^(−4p/D)、rotate-half 配对、CLS/reg pos=0 恒等）。坐标以 2^21 量化进 int32 位置 + freq_factors 抵消尺度（角度误差 <1.5e-6 rad，低于 F32 cos/sin 精度）。roll+mul+mul+add×2+cast 链（6 kernel）→ 1 个 rope kernel，CPU/CUDA 双后端同享
3. **fattn 输入零物化**：rope 补 F16→F32 跨类型输出（Q 需 F32）；Q/K/V 以 permuted **view** 直供 flash attention（mma-f16 kernel 按行步长读，无需 contiguous 物化），每层 3 个 CPY kernel 全消；CLS/reg 前缀预展开为常量权重（去掉 REPEAT）
4. **cublasLt GEMM epilogue 融合**：MUL_MAT+ADD(bias)+UNARY(GELU_ERF) 模式 → 单次 cublasLtMatmul（BIAS / GELU_BIAS epilogue，F32 累加器内加 bias）。每层 4 个 bias ADD 全消 + gelu pass 消失（GELU_BIAS 是 tanh 近似，`GGML_CUDA_LT_GELU=0` 可关；实测 sim 偏移 ≤0.0004，默认开）

**结果（RTX 5080，GPU-only 纯重放稳态）**：

| 档位 | 改动前 | 改动后 | 提升 |
|---|---|---|---|
| 256² B32 | 14.78 ms → 2165 img/s | **6.50 ms → 4921 img/s** | **2.27×** |
| 518² B4（现最优 B8） | 7.98 ms → 502 img/s | **7.67 ms → 1044 img/s** | **2.08×** |
| 224²（最优 B48） | ~2108 img/s（档位记录） | **6524 img/s** | **~3.1×** |
| 单图延迟 518² B1 | ~2.6 ms | **1.49 ms** | 1.75× |
| 单图延迟 256² B1 | ~1.1 ms | **0.57 ms** | ~1.9× |

B 扫描新最优：256²→48（4659，vs B32 +2%）、518²→8、224²→48。端到端（bench e2e，progressive 测试图集）：256² 724 img/s（解码墙）、518² B8 452 img/s（此前 351，+29%）。融合后 GEMM 有效算力 ~99 TFLOPS ≈ 5080 F16 tensor 峰值（~113）的 88%——**GEMM 本身已达硬件极限**。

**精度门（Gate B，dumpfeat 新旧对比 518²/256²×5 图）**：全特征余弦 ≥0.999992、CLS 余弦 ≥0.999985（门 0.999）；bench [sim] 偏移 ≤0.0002、[sim-cls] ≤0.0005（门 0.001）；CUDA 与 GGML_VIT_FORCE_CPU 双路径一致（≤0.0002）；oracle PASS；cargo test 双配置 7/7+1/1。

**残差 ADD 融入 GEMM（beta=1+C=residual）已实现但常态不触发**：gallocr 会把已死的 fc1-gelu 缓冲复用给输出 tensor，src1 与 dst 别名时安全检查正确拒绝（防读写冲突）；属无害保留。

**剩余构成（256² B32，6.5 ms）**：融合 GEMM ~2.2（硬件地板）+ fattn 1.17 + rope 0.83 + 残差 ADD 0.76 + proj/conv GEMM 0.76（proj 的 src1 仍 F32，fattn 输出改 F16 可再省 ~0.3，需动 fattn kernel 模板）+ norm 0.35。下一档收益需 fattn F16-out / add+norm 融合 / MMQ int8（改权重格式，精度另立门）。

### 2026-09-30 Oracle 回归门（examples/oracle.rs，新增）

新旧预处理流水线对比（旧实现内联复刻作参照），张量/特征/相似度三层，4 组配置（518/256 × progressive/baseline）全部 PASS：

- 张量级：mean|Δ| 0.12–0.19 u8 级（门 0.5），max|Δ| ≤10.2 级（门 12，硬边缘振铃），>2 级像素 ≤0.016%（门 1%）
- 特征级：自余弦 0.99967–0.99995（门 0.9995，按应用指标标定：对应成对相似度偏移 ≤0.0006）
- 相似度级：|Δsim| ≤ 0.00054（门 0.005）

注：首个版本的 oracle 有两处自身错误被它自己抓出来——`cosine_similarity` 是纯点积需先 `l2_normalize`；特征余弦阈值初设 0.9999 无标定依据，已改为按应用指标（相似度偏移）反推的 0.9995 并把依据写进注释。

改动文件：`src/preprocessing.rs`（fast_image_resize SIMD Lanczos3）、`Cargo.toml`（+fast_image_resize v6）、`examples/bench.rs`（新增：阶段拆分基准，`BENCH_TEST_DIR` 可换图集）、`examples/mkbaseline.rs`（新增：baseline 重编码）。单元测试 7/7 通过（cuda/无 cuda 两配置）。

# ViT padding NaN 定位记录

日期：2026-09-16。定位依据：用户提供的 `ViT_Padding_NaN_Investigation_Plan.md`。

## 当前结论

已撤掉诊断分支中的 attention 输出清零，并稳定复现 E-only 大图 NaN。已将有效区域的首次污染定位到 ViT block 2 的 attention（0-based），并用相同窗口长度元数据独立复现。机制是 FA3 在 Hopper TMA 路径加载逻辑序列之外、但仍在物理 tensor 内的 V 尾块；score 的长度 mask 将对应概率置零，但 PV 的 `0 × NaN` 仍产生 NaN。

这不是“CUDA Graph 必然错误”，也不是“eager 可以隔离 NaN”。共享 graph pool 的未写输出 padding 提供 NaN 来源；FA3 的 V 尾块行为把它传播到有效区域。无需模型或 graph 的两条真实序列实验也能复现。FA2 对照没有此现象。

本次完成复现与机制定位，没有提交新的 FA kernel 修复。原先输出清零是经过验证的来源阻断措施，不等于修复 FA3 的跨序列 NaN 隔离。正式 PR #56922 未修改。

## 源码、环境和复现入口

- 正式分支基线：`a6fa57b21a2771a5997789ee73baeb807b437b3f`。
- 隔离诊断分支：`codex/vit-padding-nan-investigation`，已推送 origin；撤掉清零的提交为 `558c9ba0da`。后续提交只增加或修正诊断脚本。
- 本地工作树：`/Users/jiangkuaixue/code/vllm-vit-nan`。
- 远端：`/home/david_cwq/jcz/sources/vllm-vit-nan`。
- GPU：H20-3e，UUID `GPU-8170be44-b01a-d0f7-fc73-f9b7b280cd6f`，driver `580.105.08`；node01，Slurm 3074，一张 GPU。
- Python 3.12，Torch 2.13.0+cu132，BF16，TP=1，无 LoRA，FlashAttention 3。
- Qwen2.5-VL-3B-Instruct revision：`66285546d2b821cf421d4f5eb2576359d3770cd3`；最终脚本同时固定 processor revision。
- 使用既有 uv 环境和预编译扩展。FA3 `.so` SHA256：`9b8e3081d793b426f0ccb1195476b571b2819e114274fd9a8014ac766d85adc4`。
- 安装版本元数据仍为 `0.1.dev20522+g9a855442d`。原始安装源码的 FA 依赖 pin 为 `06bdd47c0d0383daf6a2ff0c418faff9c6da16e5`，本次源码基线 pin 为 `506341a143fcabd4bb79052a7605ada727d6b3f5`；相关 TMA load/mask 路径已检查。没有重新构建 FA，不能据此声称已验证所有新版本二进制。
- 原始日志：`/home/david_cwq/jcz/artifacts/vit-nan-investigation`；本目录保存日志副本。

```bash
cd /home/david_cwq/jcz/sources/vllm-vit-nan
export PYTHONPATH="$PWD" PATH="$PWD/.venv/bin:$PATH"
export OMP_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1  # 本环境已有固定 revision 缓存

.venv/bin/python diagnostics/model_probe.py
.venv/bin/python diagnostics/fa_boundary.py
.venv/bin/python diagnostics/fa_boundary_geometry.py
.venv/bin/python diagnostics/minimal_fa3.py
.venv/bin/python diagnostics/fa_window.py
PROBE=fine .venv/bin/python diagnostics/model_probe.py
PROBE=focus .venv/bin/python diagnostics/model_probe.py
CONTROL=freshpool .venv/bin/python diagnostics/model_probe.py
CONTROL=zero .venv/bin/python diagnostics/model_probe.py
CONTROL=fa2 .venv/bin/python diagnostics/model_probe.py
SINGLE_BUDGET=1 .venv/bin/python diagnostics/model_probe.py
RUNNER=full VLLM_USE_V2_MODEL_RUNNER=0 .venv/bin/python diagnostics/model_probe.py
RUNNER=full VLLM_USE_V2_MODEL_RUNNER=1 .venv/bin/python diagnostics/model_probe.py
```

以上 Python 命令应在 Slurm 分配内执行。诊断脚本在分支 `diagnostics/`，不是生产代码。

## 1. 原始失败与对照

固定请求顺序：224×224 → 1280×720 → 224×224。同一图像资产、seed=0。实际 grid 分别为 `[1,16,16]`、`[1,52,92]`、`[1,16,16]`；有效 patch 数 256、4784、256；输出 token 数 64、1196、64。

默认九档：`[64,128,256,512,1024,2048,4096,8192,16384]`，由大到小捕获；max_batch_size=64，max_frames_per_batch=0。命中档位为 64、2048、64。

| 条件 | 小图 → 大图 → 小图 | 与普通 eager 的比较 |
|---|---|---|
| 默认共享 pool，三次全新进程 | 有限 → 全部非有限 → 有限 | 小图 max error=0；大图 NaN；三次一致 |
| 普通 eager / 同 padded buffers 的 eager | 全部有限 | 默认三场景 max error=0 |
| 每档独立 pool | 全部有限 | 三场景 max error=0 |
| 共享 pool + 输出清零 | 全部有限 | 三场景 max error=0 |
| 共享 pool + FA2 | 全部有限 | 与同 backend eager 三场景 max error=0 |
| 单档 2048 | 全部有限 | 大图 max error=0；小图 max error=1.375，不能记为完整正确性通过 |

单档小图与 padded eager 完全一致，但二者均偏离普通 eager。这是额外的数值差异观察，尚未定位，不能与大图 NaN 混为一个问题。日志中的 `mismatch` 比较对 NaN 会返回 false，因此绝不使用该字段单独判断通过；这里以有限性和最大误差判断。

原始三进程没有模型插桩。前两次独占本任务 GPU 执行；第三次末段与独立 attention/后续诊断进程重叠，因此不把本次任何并行运行的耗时用作性能数据。

证据：`model-base-{1,2,3}.log`、`model-freshpool.log`、`model-zero.log`、`model-fa2.log`、`model-single.log`。

## 2. 单 attention 边界隔离

固定有效 Q/K/V，heads=16、head_dim=80、BF16，8192 物理行、4784 有效行。padding 分别为零、有限随机值、仅 Q NaN、仅 K NaN、仅 V NaN、三者 NaN。独立参考采用每条有效序列的 PyTorch SDPA。

- FA3：零、随机、Q-only、K-only 均正常；V-only 或全部 NaN 时有效输出污染，eager 与 graph 一致。
- FA2：全部六种条件下有效区域正常。
- 预填输出尾部 NaN 后，两个版本都会保留未写 output padding。这项事实独立于有效区域是否被污染。
- 初始长序列对照中正常 FA3 max error=0.0009765625，FA2=0.00048828125。

使用真实模型记录的 `cu_window_seqlens`（577 项，末尾重复 4784）、max_seqlen=64 后：

- FA3 V-only NaN：恰好污染 **112 个有效 patch，位置 4672..4783**；eager 与 graph 一致。
- Q-only/K-only/有限 padding：有效输出正常；FA2 所有条件正常。
- 每次输出预先清零，所以这不是读取未初始化输出导致的假阳性。

进一步将输入改成两条真实序列：80、64，物理长度144，cu=`[0,80,144]`，max_seqlen=80；仅第二条的 V 设 NaN。FA3 第一条输出非有限，FA2 正常。这满足总长度等于 cu 末值，不依赖额外未声明 padding，也不需要 graph。零初始化输出仍然不能修复这个输入污染问题。

证据：`fa-boundary.log`、`fa-geometry.log`、`fa-window.log`、`minimal-fa3.log`、`minimal-fa3-zero-out.log`。

## 3. 模型首次有效污染点

探针使用预分配的整数状态 buffer 和 Triton reduction，仅存有限性计数，不保留中间 tensor 引用，不 clone activation，不在 capture 内读取 host scalar。有效 patch 前缀基于实际 replay metadata；已检查 window_index 的有效前缀是正确排列。merger 后不沿用 patch 索引。

细粒度插桩后，小图仍正常、大图仍全 NaN、小图恢复正常。0-based block 记录如下，数字是非有限元素数：

| 边界 | 有效区域 | padding 区域 |
|---|---:|---:|
| block 1 attention 输入 Q/K/V | 0 | 0 |
| block 1 attention 输出 | 0 | 3834 |
| block 1 projection 输出 | 0 | 1090560 |
| block 2 Q/K/V（每个） | 0 | 1090560 |
| block 2 attention 输出 | **143360 = 112 × 1280** | 1090560 |
| block 7 attention 输出 | **6123520 = 4784 × 1280** | 非有限 |

因此，“padding 首次非有限”与“有效区域首次非有限”发生在不同算子调用：block 1 attention 留下未写 padding，block 2 attention 才将其传入有效区域。后续 local window 扩散，block 7 全局 attention 扩散至全体有效 patch。

此为插桩运行定位，不能宣称插桩完全无扰动；重要的是失败保持存在，且独立 attention 用相同窗口元数据得到完全相同的112个受污染 patch。最初过度专门化的探针因编译开销停止；另一次诊断日志没有过滤可选 None tensor，在首个小图后报错。这两次不计入结果。最终细粒度证据为 `model-fine-v3.log`。

## 4. 内核机制与修复位置

检查的 FA3 源码路径：

- `hopper/tile_size.h`：非 local、head_dim<=96 的 BF16 路径使用 K/V tile N=144。这里模型的 window attention 已拆成 varlen 序列，kernel 的 local flag 仍为 false。
- `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`：TMA V 描述符使用物理 tensor shape；TMA 分支加载整 tile，没有像非 TMA 分支那样应用逐序列 V 加载掩码。
- 同文件对 QK score 应用 seqlen mask，再 softmax，再做 PV。mask 使越界逻辑位置的 P 为0，并不把读入的 V NaN 变成0；IEEE `0 × NaN` 仍为 NaN。
- FA2 的 `csrc/flash_attn/src/flash_fwd_kernel.h` 在最后 V tile 使用 `Clear_OOB_MN=true` 清除被谓词排除的行，与实验对照一致。

单 NaN 行位置扫描吻合144边界。例如 n=4736 时下一个边界4752，污染 n..n+15 会触发，n+16不触发；n=4864 时边界4896，污染 n+16触发，n+47不触发。真实窗口最后四条序列的起点为4672、4704、4736、4768，其144行加载覆盖 NaN 尾部，合计32+32+32+16=112个受影响 query patch。

这里读取仍在物理 tensor 内，不是已证明的非法显存越界。没有观察到必须由“同时存活的 tensor 被 graph 互相覆盖”才能解释的现象；独立 eager attention 已足以重现数值传播，故无需把共享 pool 生命周期错误当成根因。

建议修复方向优先针对 FA3 的逻辑序列尾部 V 加载/清零，确保无效 V 不参与 PV。原 wrapper 输出清零可以阻止上游未写 padding 成为 NaN 来源，但不能保证任意合法分批输入的序列隔离，也对普通 eager 添加清零成本。不能仅因其通过就称为 FA3 内核根因修复。

## 5. 影响范围和剩余验收

| 路径 | 本次自然复现 |
|---|---|
| V2 E-only | 三次原始进程均复现；细粒度插桩也复现 |
| 普通 V2 | 本次测试未复现，三场景与 eager max error=0 |
| V1 | 真正启用 encoder capture 后，本次未复现，三场景与 eager max error=0 |
| 独立 FA3 eager / graph | 人为控制 V 尾部 NaN 后均复现 |

V1 的 decoder graph NONE 会让 capture_model 提前返回，encoder manager 为空；最初该诊断报错不属于数值结果。最终 V1 用 FULL + decoder capture_sizes=[1] 激活真实 capture。普通 runner 验证关闭无关 decoder compilation/JIT warmup，但保留实际 encoder graph；没有进行完整文本生成质量评测。未复现不证明 V1/普通V2不受影响，显存内容和分配历史不同即可改变自然触发。

本轮是定位，不是新修复的发布验收。未修改或构建 FA kernel，没有新增修复后模型全套回归、输出生命周期/EC端到端验收或性能对比；不能把以前输出清零方案的性能数据套用于尚未实现的内核修复。单档2048的小图偏差另需定位。

## 最终复核与清理

最终诊断代码提交：`394ef75a4b`。进一步缩小到 block 1/2 的20个算子检查点，独立进程再次复现完全相同计数：block 1 attention 的3834个 padding 非有限值全部为 NaN；block 2 attention 输出的143360个有效区域非有限值也全部为 NaN。其有效 Q/K/V 在调用前仍为有限值。证据：`model-focus.log`。因此首次污染结论在两种探针密度下保持一致。

最终模型及 attention 实验均已结束，nvidia-smi 没有本任务 GPU 进程；Slurm 3074 已释放。没有停止其他用户的作业。

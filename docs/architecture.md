# VisionForge 架构文档

## 1. 系统定位
CPU-only、零下载可运行的内容图像检索（CBIR）系统。核心差异化 = **AMDF 自适应多描述子融合**。
所有检索能力复用业界顶级开源（opencv / scikit-image / pillow / numpy），无自研 SOTA 部分。

## 2. 模块划分与接口契约

### core/（无业务依赖，被所有人依赖）
- `types.py`：`ImageRecord` / `RankedMatch` / `QueryResult` / `BenchmarkResult`
- `errors.py`：错误码 E100~E500（Config/Data/Descriptor/Index/Pipeline）
- `config.py`：`Config` dataclass，支持 `ENV_VISIONFORGE_*` 覆盖
- `interfaces.py`：`Descriptor` / `SimilarityIndex` Protocol（运行时契约）

### domain/（领域能力）
- `registry.py`：探测 opencv / scikit-image / pillow 可用性（离线兜底依据）
- `descriptors.py`：5 个描述子，统一输出 **L2 归一化固定长度向量**，余弦相似度=点积
  - `color_hist`（永远可用，numpy/pillow）
  - `hog` / `lbp`（scikit-image）
  - `orb` / `sift`（opencv）
  - 无关键点时确定性兜底向量，保证非零、等长、可复现

### preprocess/（预处理）
- `transforms.py`：resize / 灰度 / 归一化（Pillow）

### data/（数据）
- `synthetic.py`：合成 shape×color 组合数据集（自带 ground truth，离线可评测）
- `loader.py`：从磁盘目录载入真实图像

### training/（索引）
- `index.py`：`ForgeIndex` 为每个描述子维护归一化向量矩阵；`per_descriptor_similarity()` 返回各描述子余弦相似度

### hpo/（融合权重学习）
- `selector.py`：`learn_weights()` 在验证集上以 mAP 为目标做坐标上升，学习非负归一化权重

### eval/（评测）
- `metrics.py`：`average_precision` / `recall_at_k` / `mean_average_precision`

### pipeline/（编排）
- `pipeline.py`：`VisionForgePipeline.build()`（建索引 + 学习权重）→ `query()` → `benchmark()`；`_fuse()` 实现 AMDF 融合

### cli.py / examples/run_demo.py
- 命令行入口与端到端演示（落盘 `benchmark.json`）

## 3. 调用关系（单向无环）
```
cli → pipeline → {data, hpo, training, domain, eval} → core
```
- 数据流：`image → descriptors(向量) → index(相似度) → pipeline(融合+排序) → eval(指标)`
- 权重流：`hpo.selector` 消费 `index.per_descriptor_similarity` + ground truth，产出 `weights` 回灌 `pipeline._fuse`

## 4. 关键不变式（手写实现验证标准）
1. 每个描述子输出 L2 范数 ≈ 1（余弦相似度等价于点积）。
2. 同一描述子对任意图像输出长度固定（矩阵可 `stack`）。
3. 查询自身相似度最高（自匹配排第一）。
4. AMDF 在权重学习的验证划分上，融合 mAP ≥ 每个单描述子 mAP（坐标上升构造性保证）。
5. 权重非负且和为 1。

## 5. 性能基线（真实运行，见 benchmark.json）
- 融合 mAP 0.959 ＞ 最佳单描述子 0.842
- Recall@5 = 0.696；查询延迟 0.030 ms；索引内存 0.71 MB

## 6. 可扩展性
- 新增描述子：在 `domain/descriptors.py` 实现 `extract()` 并加入 `build_descriptors()`，自动进入融合。
- 切换深度描述子（如 CLIP）：同样实现 `Descriptor` 接口即可，无需改动 pipeline/eval。
- 大规模索引：实现 `SimilarityIndex` 接口替换 `ForgeIndex`（如 FAISS 后端）。

# 投稿前最终撞车扫描报告

**扫描日期**：2026-08-01
**扫描范围**：arXiv（cs.CV / cs.CL），重点 2025-06-01 至今的新文与新版本
**方法**：arXiv 官方 API，多组关键词组合查询（每组取按提交日期倒序前 40 条），逐条按摘要判定与两篇论文主张的重叠度；另按 id 精查六个基线论文的最新版本。

**被扫描论文**
- **审计论文**《How Fragile Is Your Vocabulary? A Controlled Audit of Inference-Vocabulary Robustness in Training-free OVSS》：冻结推理协议，仅沿同义词替换/粒度偏移/干扰类注入三轴扰动词表；五个发现（官方词表工程价值最高 20.6 mIoU、同义词替换成本与种子方差、干扰类注入的背景欠建模机制、whitening 修复的几何—分割解耦、自动化词表工程的可恢复/不可恢复分解）。
- **方法论文**《REVA: Region-Evidence Arbitration with Vocabulary-Adaptive Background Synthesis》：训练无关插件 = VABS（facility-location 选取背景负例 + 相似度安全过滤）+ SAM automask 区域证据仲裁；plain 词表下 +19.9~+23.1 mIoU，自攻击表明大部分文本侧收益是背景 logit 再校准。

**总体结论：未发现严重撞车。** 两处需主动划界的**部分重叠**：SynCLIP（CVPR 2026，同义词鲁棒性但走预训练路线）与 ActiveSAM（2026-06，图像条件类剪枝，与词表干预空间相邻）。其余命中均为方法改进类或异域（遥感/医学/3D）工作，与"受控审计"与"词表自适应背景合成+区域仲裁"的组合主张不冲突。

---

## 一、各组查询命中列表与重叠度评级

评级说明：**无重叠** = 主题相邻但主张不冲突；**部分** = 共享一个子主张或动机，需在 related work 划界；**严重撞车** = 核心主张/方法/实验设计实质相同（本次扫描未发现）。

### Q1. open-vocabulary segmentation vocabulary robustness / class name sensitivity

| 命中（2025-06 起） | 日期 | 评级 |
|---|---|---|
| **SynCLIP: Synonym-Coherent Language-Image Pretraining for Robust Open-Vocabulary Dense Perception**（arXiv 2607.11008，CVPR 2026） | 2026-07-13 | **部分** |
| ViP²-CLIP（2505.17692v3，异常检测） | v3 2025-10-06 | 无重叠 |

**SynCLIP 分析**：确认了"同义词导致 grounding 不一致"这一现象，与审计论文发现 (2) 的动机直接相邻。但它是**预训练/训练方案**（提出 Synonym-Coherent 预训练框架来修复），而审计论文是**推理词表的受控扰动测量**（训练无关、冻结协议、含粒度/干扰/种子方差等 SynCLIP 未量化的轴），REVA 是**推理期插件修复**。属于"问题相邻、路线正交"，必须引用并划界（见第三节）。

### Q2. prompt robustness / prompt sensitivity segmentation

| 命中（2025-06 起） | 日期 | 评级 |
|---|---|---|
| Prompt Group-Aware Training for Robust Text-Guided Nuclei Segmentation（2603.06384，医学） | 2026-03-06 | 无重叠 |
| On the Robustness of 3D Medical Segmentation Models Against Imprecise Visual Prompts（2601.16383） | 2026-01-23 | 无重叠（视觉提示，非文本词表） |
| Beyond Templates: Revisiting Zero-Shot RS through Meta-Prompting（2606.20702） | 2026-06-15 | 无重叠（遥感分类模板敏感性；可作旁证引用） |
| PixFoundation v4（2502.04192，pixel-level VFM 评测方向反思） | v4 2026-01-24 | 无重叠 |

### Q3. negative queries / background segmentation

| 命中（2025-06 起） | 日期 | 评级 |
|---|---|---|
| NegROI: Scene-Conditioned Negative Prompts for Interactive 3D Segmentation（2607.05955） | 2026-07-07 | 无重叠（3D 点云交互分割、点击负提示，非文本词表负例） |
| Bridging Semantics and Geometry: LVLM-SAM for RS Reasoning Segmentation（2512.19302v2） | v2 2026-04-21 | 无重叠 |
| SADL: Subject-Aware Distractor Localization benchmark（2606.30393） | 2026-06-29 | 无重叠（摄影构图中的视觉干扰物定位，与"干扰类注入"仅词面相似） |

### Q4. SAM region pooling / SAM + CLIP training-free

| 命中（2025-06 起） | 日期 | 评级 |
|---|---|---|
| TraceCLIP（2607.26107） | 2026-07-28 | 无重叠（patch-to-CLS 归因恢复局部语义，纯 CLIP 侧） |
| SegEarth-OV3 / OmniOVCD / Prompt-Calibrated SAM 3 等一批 **SAM 3 + OVSS（遥感）** 工作（2512.08730、2601.13895、2606.21863 等） | 2025-12 起 | 无重叠（异域；但注意 SAM 3 概念提示范式正在快速铺开，见第四节建议） |
| ARM: Learnable Plug-and-Play Module for CLIP-based OVSS（2512.24224） | 2025-12-30 | 无重叠（可学习模块，非训练无关） |
| Plug-in Feedback Self-adaptive Attention in CLIP for Training-free OVS（2508.20265） | 2025-08-27 | 无重叠（注意力侧插件，不涉词表/背景负例/区域仲裁） |
| DouC / OV-Stitcher / PEARL / ConInfer / NERVE / LPOSS 等 2025-06 后训练无关 OVSS 新方法（十余篇） | 持续 | 无重叠（均为提升官方协议 mIoU 的方法改进，未研究词表鲁棒性；反而扩大了审计论文"可复跑协议"的适用对象） |
| ReME: Data-Centric Framework for Training-free OVS（2506.21233，参考集质量视角） | 2025-06-26 | 无重叠 |

### Q5. detector-guided vocabulary pruning / class pruning

| 命中（2025-06 起） | 日期 | 评级 |
|---|---|---|
| **ActiveSAM: Image-Conditional Class Pruning for Fast and Accurate Open-Vocabulary Segmentation**（2606.16996，SAM 3 backbone，preprint） | 2026-06-15 | **部分** |
| Training-Free Class Purification for OVSS（FreeCP，2508.00557，ICCVW） | 2025-08-01 | **部分**（REVA 已引用并划界："pruning 减类 vs VABS 加负例"——保持该表述即可） |

**ActiveSAM 分析**：训练无关、图像条件地从数据集词表中剪出"活跃类子集"再解码，动机含类冗余（与审计论文的干扰类注入实验、REVA 的负例词表处在同一个"推理词表干预"空间）。差异清晰：ActiveSAM 是**做减法提效/提精**（且基于 SAM 3 概念提示范式），REVA 是**做加法建模背景**并证明其机制是背景再校准；审计论文测的是**方法对扰动的响应**而非提出干预。建议与 FreeCP 一并归入"词表剪枝"一类划界。

### Q6. benchmark naming bias / renaming / label-name perturbation

2025-06 后无新命中与两文冲突（命中多为 LLM 方言歧视、机器文本检测等无关工作）。RENOVATE（2403.09593）仍是最近邻，双方论文已划界。

**Beyond Standard Benchmarks: A Systematic Audit of VLM's Robustness to Natural Semantic Variation（2604.04473，2026-04-06）**：也自称"systematic audit"，但对象是 VLM 分类/检索任务在排版攻击、自然语义变体下的鲁棒性，不做密集分割、不扰动推理词表。**无重叠**，可在审计论文 related work 补一句以防审稿人联想。

### Q7. background synthesis / contrastive concepts

| 命中 | 日期 | 评级 |
|---|---|---|
| TCC v3（2407.05061v3，TMLR camera-ready） | v3 2025-06-16 | **部分**（已知近邻；v3 无新增撞车内容，见第二节） |
| VocaDet（2607.08541，检测+向量库） | 2026-07-09 | 无重叠 |

### Q8. 其他补充查询（vocabulary-free、granularity/hypernym、facility location、whitening/anisotropy、audit robustness VLM）

- vocabulary-free 方向 2025-06 后新文集中在细粒度识别/少样本分类（2512.18897、2507.23070、2506.04005），无分割词表审计。**无重叠**。
- granularity/hypernym：GUIDED（2603.27014，细粒度 OV 检测）与审计论文的粒度轴仅词面相近。**无重叠**。
- facility location + CLIP/vocabulary：无相关命中——**VABS 的 facility-location 负例选取组合仍然独占**。
- CLIP 文本嵌入 whitening/anisotropy + 分割：无命中——审计论文发现 (4)（几何—分割解耦）**未被抢发**。

---

## 二、六个基线的版本与后续工作核查

| 论文 | arXiv id | 最新版本 | 状态与影响 |
|---|---|---|---|
| Open mIoU（Rethinking Evaluation Metrics of OVS） | 2311.03352 | **v1（2023-11-06，无更新）** | 无新版本、未见直接后续。审计论文引用现状不受影响 |
| FLOSS | 2504.10487 | **v2（2025-07-30）**，标注 **ICCV 2025** | camera-ready 更新；仍是"每类模板选择"，未扩展到词表扰动审计。引用时可更新为 ICCV 2025 |
| RENOVATE（Renovating Names in OV Segmentation Benchmarks） | 2403.09593 | v2（2024-05-24，无新版本） | 未见续作。划界表述可保持 |
| Neglected Tails in VLMs | 2401.12425 | v3（2024-05-22，无新版本） | 无变化 |
| Trident | 2411.09219 | **v1（2024-11-14，无更新）** | 无新版本；REVA 的同协议 anchor 引用不受影响 |
| TCC | 2407.05061 | **v3（2025-06-16）**，标注 **TMLR camera-ready** | 2025-06 有版本更新（期刊定稿）。REVA §5 已声明"TCC 对比留作未来工作"——TCC 现为正式发表，建议投稿版把该声明保留并把引用更新为 TMLR |

---

## 三、划界建议（针对两处部分重叠）

### 1. SynCLIP（CVPR 2026）↔ 审计论文发现 (2) / REVA 动机

- **审计论文**：在 related work 的 "Text-side sensitivity" 段加一句，模式沿用现有划界逻辑——SynCLIP 通过同义词一致的**再预训练**修复 grounding 不一致，属于"改模型迁就名字"；我们**冻结模型、扰动名字**，量化训练无关方法族的响应（含 SynCLIP 未涉及的粒度、干扰注入、种子方差轴），二者互补且我们的协议可直接用于审计 SynCLIP 类模型。
- **REVA**：在 related work "Vocabulary robustness" 段补一句：SynCLIP 用预训练获得同义词鲁棒性，需要重新训练编码器；REVA 是纯推理期插件，且针对的缺口（plain-vs-engineered 词表差距、背景欠建模）SynCLIP 不处理。
- 可加分点：审计论文 §6 讨论"未来审计对象"时点名 SynCLIP 作为可检验的修复路线。

### 2. ActiveSAM（2026-06）+ FreeCP ↔ REVA / 审计论文干扰类实验

- **REVA**：把现有"FreeCP and vocabulary pruning works subtract classes, whereas VABS adds negatives"一句扩为同时涵盖 ActiveSAM（图像条件剪枝、SAM 3 概念提示），强调二者正交可叠加：剪枝减少假阳性竞争，VABS 补足背景建模；且我们的自攻击揭示的"背景 logit 再校准"机制是剪枝类方法不提供的解释。
- **审计论文**：在讨论干扰类注入结果时可加一句：新出现的图像条件类剪枝（ActiveSAM、FreeCP）隐式假设"词表越准越好"，而我们的结果（注入干扰反而抬高 GT-mIoU、机制在背景欠建模）说明该假设的收益方向依赖背景建模与度量约定——这为剪枝类工作提供了审计维度，非竞争关系。

### 3. 低成本防御性补引（可选）

- Beyond Standard Benchmarks（2604.04473）：审计论文 "Evaluation audits" 段一句话区分（分类/检索层面的语义变体审计 vs 密集分割词表审计）。
- Meta-Prompting RS（2606.20702）：作为"文本设计敏感性在其他域同样成立"的旁证。
- SAM 3 概念提示生态（SegEarth-OV3 / ActiveSAM 等）：REVA 限于 SAM ViT-B automask；若审稿人问 SAM 3，可在 limitation/future work 提一句"区域仲裁与概念提示解码的结合"。

---

## 四、附：查询覆盖记录

批次 1（10 组）：vocabulary robustness、class name sensitivity、prompt robustness segmentation、negative/background queries、SAM+CLIP region pooling、vocabulary pruning、benchmark naming bias、synonym robustness、training-free OVSS 全量、background synthesis/negatives。
批次 2（12 组）：Open mIoU、FLOSS、RENOVATE、Neglected Tails、Trident、TCC 后续、class pruning/purification、detector-guided vocabulary、vocabulary expansion 评测、evaluation bias、CLIP whitening/anisotropy、class-name engineering。
批次 3（9 组）：Trident-OVSS、vocabulary-free、granularity/hypernym、distractor classes、background class、SAM automask pooling、facility location、VLM robustness audit、label-name perturbation。
另按 id 精查 6 个基线论文最新版本与 comment 字段（会议/期刊状态）。

局限：仅覆盖 arXiv 公开条目；同期在审未挂 arXiv 的工作无法检出。arXiv API 的短语匹配偏保守，已用多组近义查询交叉覆盖降低漏检率。

# 收官前终审报告：claims-evidence 对账 + 全局一致性清扫

核验基础：审计论文 PDF、REVA 论文 PDF 全文提取 + RESULTS.md（stage-12 全月记录）逐条数字复算。未覆盖处已注明"早期 stage、无法从所给 RESULTS 复核"。

---

# 一、必改（Must-fix，共 8 项）

**M1. 运行数自相矛盾：摘要 "about 150 evaluation runs" vs §6 Limitations "the ∼130-run matrix"。**
这是旧稿残留（摘要已更新、Limitations 未更新），两处必须统一为同一个数（按最终矩阵重数）。

**M2. 摘要 "The audit yields four findings" 但随后列出 (1)–(5) 五条。**
数字改为 five（或合并两条）。

**M3. §4.3 正文 vs Figure 3 caption 数字不一致。**
正文说 "+200 near condition collapses to ∼5 mIoU"，caption 说 "collapses to ∼4"。RESULTS（W3b/W4a）all-class 为 3.5–4.3 → 统一为 ∼4。

**M4. §4.6 LLM judge 范围错误。**
"worsens every damage arm (−3.3 to −4.5)"。RESULTS W11-J2 显示 freqctrl 臂只降 −0.6/−0.9，不在该区间。改为 "−0.6 to −4.5"，或把区间明确限定到 syn100/ANS 臂。

**M5. Limitations 的全量分割校准数字与 Table 6 不一致且未声明分割。**
Limitations 说 "plain 35.2/34.5/32.5 vs official 56.9/55.4/44.4"（SCLIP/ClearCLIP/MaskCLIP），而 Table 6（W4a，dev-excluded 1349 张）是 35.6/34.8/32.8 vs 57.2/55.7/44.6。若 Limitations 用的是含 dev 的 1449 张旧校准，必须标注分割（1449 vs 1349），否则读者会视为矛盾；该组数字也不在本月 RESULTS.md 中，需回溯到 run 记录确认来源。

**M6. REVA 摘要/§4.2 "boost reproduces 75–85% of VABS's text-side gain" 下界错误。**
按 Table/W13-L5 复算：SCLIP (50.7−34.7)/(53.5−34.7)=85.1%，NACLIP (48.4−36.6)/(52.8−36.6)=**72.8%**，下界不是 75。RESULTS.md 本身也写 75-85（同样错）。改为 "≈73–85%" 或 "roughly three quarters to 85%"。注意此数字在审计论文 §4.3 机制探针互引处同源出现，**两篇需同步改**。

**M7. 审计 §4.4 "The only consistently positive cases are small-vocabulary VOC-21 (+0.9 to +1.5 for ZCA)" 无任何表格/附录支撑。**
Table 4 无 VOC-21 列。需加表格引用或附录指针，否则属 (a) 类无证据主张。

**M8. 两篇 COCO-Object plain 基线互相矛盾且分割未声明。**
REVA §4.4 说 "full val2017, plain 22.4/22.4 (SCLIP/NACLIP)"；审计 Table 5 COCO 块为 SCLIP 22.6、NACLIP 22.1（caption 只对 VOC 声明 test-300，COCO 块分割未声明）。两值均不在 RESULTS.md（早期 stage）。需要：
1. 审计 Table 5 caption 补 COCO 分割声明；
2. 核对 REVA "22.4/22.4"（两法完全相同较可疑，疑似笔误）；
3. 若分割不同，需在互引处注明，避免读者对同名 protocol 见到两组 plain 值。

---

# 二、审计论文 claims-evidence 对账表（摘要 + 引言 + 结论）

| 主张 | 证据位置 | 判定 |
|---|---|---|
| ~150 evaluation runs | §6 Limitations 说 ∼130 | ✗ 见 M1 |
| naming 工程 up to 20.6（11.8–20.6 across methods） | Table 1 (test-300)；RESULTS W3b NEG 11.8/20.7/18.7 | ✓（但 Table 6 全量分割 NEG 达 21.6–22.1；摘要为 test-300 三法口径，建议注明） |
| synonym substitution up to 6 mIoU | Table 2（ClearCLIP 5.9 最大） | ⚠ NACLIP 6.8（§5.1）、ViT-L 7.0（§5.2/W1a）超出 "up to 6"；建议改 "up to 6（主三法）；复制中至 7" |
| synonym seed spread 3.7 mIoU | Table 2：SCLIP 34.4 vs 30.7 | ✓ |
| hypernym halves VOC-21 / nearly zeroes ADE-150 | Table 2：34.8→17.0；13.1→2.5 | ✓ |
| distractor 提升 up to 10 / classification 控制 −18.5 | Table 3（34.8→44.5=+9.7）；72.8→54.3 | ✓；§4.3 已声明单细胞控制（SCLIP/VOC-21/+200near，GT-box crops），Limitations 也声明"single cell"；摘要处未逐字声明，可接受 |
| engineered 词表下反转为降（1.7–2.9） | Table 3 底行：−1.7 至 −2.9 | ✓ |
| 文本侧修复 8/9 cells 降 mIoU | Table 4（centering 8/9，ZCA 9/9；例外 = centering 助 SCLIP PC-459 +0.9） | ✓（摘要措辞与 centering 一致） |
| VABS up to +18.8 / macro +16.1 / random +12.6 / 选择优势 +3.5 / 达官方 96–102% | Table 5，逐格复算全部吻合（macro 16.05；rand 12.55；选择 3.5） | ✓（ClearCLIP/NACLIP 实为 95.8–95.9%，四舍五入到 96 勉强成立，见 R6） |
| COCO-Object 转移 +5.0 至 +10.7、macro +8.9 | Table 5 右块，复算吻合（9.2/10.6/5.0/10.7 → 8.875） | ✓（分割声明缺失，见 M8） |
| 引言 2(c)：ClearCLIP 领先 SCLIP 2.5（PC-459）/ 落后 1.8（VOC-21 官方） | Table 4 none 列（15.3 vs 12.8）、Table 1（55.4 vs 53.6） | ✓ |
| 引言贡献 5 "resisted **three** pre-registered label-free mechanisms" vs 结论 "**every** label-free mechanism" | 附录 graveyard 实为 ~7 个信号族 | ⚠ "three" 疑为旧稿残留，与结论口径不一，建议统一（见 R3） |
| 结论各定性句（naming 工程可比拟方法差、synonym 噪声大于典型增益、mIoU 对扩表的响应是 bg 建模+度量约定 artifact、几何修复助诊断不助分割、per-class 保护抗拒所有 label-free 机制） | §4.1–§4.6、附录 C（W5e）、Table 8 kill 表 | ✓ 全部可落位 |

**§5 复制/扩展章节数字核验结果**（对 RESULTS.md 逐条）：
NACLIP 复制（55.0/36.5、+5.9/−2.7）、ViT-L（37.2→30.2、36.3→30.7、NEG +3.5/+14.3、distractor −1.1/+5.6、all-class 3.4/4.0）、Table 6 全量分割全表（W4a 逐格一致）、ANS（13.4/13.7 vs 随机最差 28.3/31.0、迁移 13.3/15.2、无搜索过拟合）、token-matched 控制（15.3/21.0=73%、OWLv2 25.3/31.8=80%、residual 5–7、控制种子 18.5–19.7）、频率律（Spearman −0.22、BPE proxy −0.12、757 观测/9 模型）、canonicalizer（恢复 35–44%、伤已规范名 3–4）、LLM judge（−3.3~−4.5，见 M4）、跨家族（GDINO −37.6/−28.3、OWLv2 −7.1/−12.9、ANS 迁移 −31.8/−38.9、OWL-ViT v1 −19.0、MDETR 3.4、MM-GDINO 32.2/22.2/24.7、distractor-fire 17.8/23.6%、92% bg、fg steal 4.5/8.5% vs 29–43%）、跨语言（es 52–62% / best-of-3 86%、84%、GDINO 18%→25%、de/ru 缺口 14.0/10.9/12.7、ru OWLv2 5.9%、zh <10 + H4 artifact 控制）、附录 B/C（ρ=0.89 vs 0.03、routing +14.9、router 53.4 vs 54.0、consistency 46.8/21.9、8-pool oracle +2.4、conformal 59.7%/72%/92.3%、spectral 0.029/97%、transplant +1.7~+12.8、presence gating 0.24@0.99 与 0.31–0.35@0.98、boost 47.5 vs 46.5、Spearman 1.000、K3 oracle ~4 与 GT-present −1.0~+6.5、K2 污染曲线 +6~+9@10）——**全部与 RESULTS.md 吻合**，且 test-300/单种子/style-reimpl/in-distribution 限定语基本到位（例外见建议 R5）。

---

# 三、REVA claims-evidence 对账表（摘要 + 引言 + 结论）

| 主张 | 证据位置 | 判定 |
|---|---|---|
| 方法 lose 12–21 mIoU（plain vs official） | 审计 11.8–20.6 / Table 6 至 21.6 | ✓（四舍五入口径） |
| REVA 提升 +19.9 至 +23.1 | Table 1 复算：+23.1/+21.3/+19.9/+22.3 | ✓ |
| closing 86–96%（per-method 90/86/96/87） | Table 1 复算：90.2/85.5/95.7/86.8 | ✓ |
| official 上再加 +4 至 +9 | Table 1：+4.0/+4.0/+9.0/+5.7 | ✓ |
| boost 复现 75–85% text-side gain | 复算 85.1% / **72.8%** | ✗ 见 M6 |
| VABS 残余优势 +2.8 至 +4.4；boost 叠加仅 +0.2 至 +1.1；b*=0.08 使 SCLIP 崩到 33 | W13-L5 精确吻合 | ✓ |
| boost 不迁移（ctx60 plain −0.3~−0.9 / VABS 上 −5.8~−9.6；ADE −1.3~−1.4；ADE plain 无 bg 类） | W13-L5t 精确吻合（−0.32/−0.88；−5.78/−9.63；−1.30/−1.42） | ✓ |
| matched compute 下官方保留 0.9–3.6 领先 | Table 1：2.5/3.6/0.9/3.4 | ✓ |
| 安全判据（vs pixel-VABS）：无类在 2+ 法上降 >3，最差 person −3.2 (NACLIP) | Table 3：42.5→39.3 | ✓，判据 scope 表述清楚（明确写的是 REVA vs pixel-VABS 而非 vs plain） |
| vs plain：person −6.0/−6.0/+8.6/−8.9；tvmonitor −2.9/−11.2/−6.3/−11.8；新基 person −7.7/−15.4/−13.0、tv −13.9/−12.0/−13.0；pottedplant +13~+18 | Table 3 + RESULTS W4h | ✓ |
| SAM alone +0.4~+6.2；仲裁 over pixel-VABS +3.2~+7.6；选择优势 +4.7~+5.9 | Table 1 复算全部吻合；单种子已在正文与 caption 声明 | ✓ |
| oracle AUC 0.92–0.97、SLIC dev −3.0、visual veto −0.08 | §3/§4.3；源于早期 stage，本月 RESULTS.md 未覆盖 | ⚠ 无法外部复核，建议 artifact 列 run 文件（R7） |
| COCO 22.4/22.4→34.4/36.3、仲裁 +2.5/+3.9、选择 +1.1/+0.8 | 不在 RESULTS.md | ⚠ 见 M8 |
| ctx60 "+2.0/+3.1 (31.2/31.8→33.3/35.0)" | 端点差值为 2.1/3.2 | ⚠ 四舍五入不自洽，见 R1 |
| ADE +1.0/+1.3（13.1→14.2、14.7→16.1） | W2a：1.04/1.32 | ✓ |
| ViT-L 六格（REVA 44.0/51.6 超过 pixel official 40.7/50.6；选择 +1.8/+3.1；ViT-B 文本嵌入迁移已披露） | W1a/W4i（+1.8 实为 1.77） | ✓ |
| ProxyCLIP-style：99%（vs pixel official 58.8）/ 85%（vs matched 62.1）、+4.2 选择 | W3a/W4e（85.4%） | ✓ |
| LPOSS-style：94% 闭合但选择仅 +1.3 低于 +2 复制线（诚实降格为"只主张 gap closure"）；SC-CLIP：89.4%、+3.1、超 pixel official 但低于 matched 2.4 | W3d/W4e 逐条吻合；旧 "86–103%" 全文 0 处（已按 matched 框架更新） | ✓ |
| author-code anchor：47.3→56.2（恢复 64%）、official 61.2、SAM 未 anchor 已声明 | W4g | ✓ |
| LaVG-style 变体 +1.2 内、hand-written bg 最多 +2.3、PAMR 62.6 vs 发表 64.1、SAM 0.42s vs 0.035s（∼12×）、Trident ∼3× 便宜 | Table 2 / RESULTS | ✓ |
| J5 剪枝：+1.4~+9.2 mean +4.4、syn 臂 30.55→35.96 也升、REVA 叠加 +2.4/+2.5/+3.6（59.4 vs 官方 pixel 58.8）、像素流 55% absent / 3% present、class-level Spearman −0.20、ADE +2.5~+2.7、granularity 探针 ρ=+0.17 反号、recall 0.67、top-k 达 80%（VOC）但 ADE −3.9、soft λ=0.3 +0.6/−1.7、ANS 下剪枝 ~+7 / ~35% 恢复（非防御）、OWLv2 synonym drop 7.1 | W11-J5 / W12-K1 / W12-K4 / W13-L2 / W13-L4 全部吻合；"preliminary single-seed, 300-image subsets" 已声明 | ✓ |
| 结论："recovers 86–96% under matched compute"、"0.9–3.6 lead"、"person/tvmonitor 残害减轻未消除"、"SAM dominates cost" | 与 Table 1/3、§4.2 一致 | ✓ |

---

# 四、多轮改写残留清扫

## 已确认清理干净
- **"fourth axis"**：两篇 0 处；现为 "cross-lingual probe: a Spanish case study"。
- **"52–62%"**：审计中两处均已带 translation-choice 解读（§5.5），且 GDINO best-of-3 25% 的"非 translation-choice artifact"限定在位。
- **lineage/血统**：§5.4 以 "falsified as the organising variable / training recipe rather than text-encoder pedigree" 出现，§5.5 有 lineage-vs-multilingual 消歧；1285 行 "Text-encoder-lineage banding" 是 Table 8 kill 表条目（记录假设被杀），恰当。REVA 中 lineage/pedigree 0 处。
- **granularity 旧解释**：REVA 已替换为 "a pre-registered granularity probe rules out the obvious explanation…the shrinkage mechanism remains open"；旧句 "plausibly because detector recall degrades on finer-grained queries" 0 处。
- **VABS 校准+选择两分**：REVA 摘要与 §4.2(iv-b) 均成型（"automatic background re-calibration…residual advantage is the selection component"）；审计 §4.3/§4.5 的 one-parameter 控制与 background-sink/per-class 两分一致。
- **REVA 中 rare/low-frequency**：0 处。

## 仍存在的残留（均在审计论文）
1. **§6 "∼130-run matrix"** —— 最硬的旧稿残留（M1）。
2. **§4.6（约 785 行）"recovers 35–44% of rare-control and searched (ANS) damage"** —— 按 W10 评审整改承诺（"all rare/low-frequency phrasing replaced with non-canonical-name sensitivity"），应改 "non-canonical-control"。
3. **§5.4（约 1015 行）"equally-rare synonyms drawn from the same candidate pool"** —— W7a 控制是 token-matched 非常规名；"equally-rare" 是已被 W9 杀死的频率框架残留，建议改 "matched non-canonical synonyms"（同段随后自己也说 "shared non-canonical-name sensitivity"，前后不一致）。
4. §5.3（约 946 行）"often low-frequency senses" 与附录（约 1122 行）"a rarer sense"：描述词义罕见、属可接受的描述性用法；若追求措辞统一可改 "non-dominant senses"。
5. **§4.3 "collapses to ∼5 mIoU"** 亦属旧数字残留（M3）。

---

# 五、两篇互引一致性（REVA 引审计 [9]）

| REVA 中的引用 | 审计原文 | 判定 |
|---|---|---|
| 引言 "official vocabularies embed undocumented engineering worth 11.8–20.6 mIoU on VOC-21" | 摘要/贡献 2(a) 同数 | ✓ |
| "no pre-registered text-only mechanism could recover the engineered value safely…systematically harmed person/tvmonitor…harmful negatives are textually distant yet visually close, which no text-similarity filter can detect" | §4.5 原话一致 | ✓ |
| §4.2(iv-b) 单参数对照："the companion audit's one-parameter test: any absorption claim must beat a direct background calibration" | 审计 §4.3 机制探针原句（"one-parameter control"）一致；boost 数字（b*=0.03/0.05、50.7/48.4、75–85%）同源（W13-L5） | ⚠ 判据一致，但**两篇同时继承 M6 的 75% 下界错误，需同步改** |
| 外部监督披露：REVA §4.6 "imports external supervision (detector pretraining)…consistent with the companion audit's finding that per-class protection tracks supervision"；boost 需 labelled dev split 在摘要/§4.2 声明 | 审计 §4.5/结论 "encode supervision that label-free machinery cannot reconstruct"；§5.4 headline 段含 in-distribution caveat | ✓ 一致、披露对称 |
| 检测器脆弱性：REVA §4.6 "the detector's own vocabulary sensitivity (synonym drop of 7.1 for OWLv2 — mild)" | 审计 §5.4 OWLv2 VOC −7.1 | ✓（建议 REVA 补一句 OWL-ViT v1 −19.0，避免读者误以为 CLIP-tower 检测器普遍温和——审计已证 lineage 分带不成立） |
| REVA "the same signal fails completely as a distractor-axis shield (see the companion audit)" | 审计 Table 8 box-shield 4.23→4.27（W11-J4） | ✓ |
| REVA §4.4 "NEG +22.1 on the full dev-excluded split" | 审计 Table 6 ProxyCLIP-style 22.1 | ✓ |
| REVA Figure 3 caption "Every class improves except person and tvmonitor" | Table 3 中 MaskCLIP person 为 **+8.6（改善）** | ⚠ 若图为四法平均（person 均值 −3.1、tv 均值 −8.1）则成立，caption 应写明 "averaged over methods"；若图分法显示则 caption 错 |

**总评**：单参数对照（判据表述、数值、labelled-dev 披露）、外部监督披露（boost/剪枝/检测器三处均声明）、检测器脆弱性（−7.1、剪枝非分布防御、box-shield 失败）在两篇间一致且互为支撑，无实质冲突。唯一共享错误是 75–85%（M6），唯一表述风险是 Figure 3 caption。审计的否定结论（text-only 负词表伤 per-class）与 REVA 的建设性主张（区域仲裁按其判据不新增伤害、但 vs plain 残害未消除）措辞互洽，REVA 未夸大为"消除伤害"。

---

# 六、建议（Recommended，共 8 项）

- **R1.** REVA §4.4 ctx60 "+2.0/+3.1（31.2/31.8→33.3/35.0）"：端点差为 2.1/3.2，统一小数位。
- **R2.** 审计摘要 "up to 6 mIoU"：建议注明为主三法口径（NACLIP 6.8、ViT-L 7.0 见 §5.1/§5.2）。
- **R3.** 审计引言贡献 5 "resisted **three** pre-registered label-free mechanisms" 与结论 "**every** label-free mechanism"、附录 graveyard（~7 信号族）口径统一。
- **R4.** 审计 §5.5 "84% on OWLv2…crossing the frozen translation-choice band"：按冻结判据 OWLv2 84.4% 在 85% 线之下（RESULTS 明写 "just under the line"），措辞应改为 "SCLIP 越线、OWLv2 同方向但略低于线"。
- **R5.** 审计 §5.1（NACLIP 55.0/36.5、+5.9/−2.7 等）未声明 test-300；§5.2/§5.5 均已声明，补一句即可。
- **R6.** Table 5 的 "96–102%"：ClearCLIP/NACLIP 实为 95.8–95.9%，可写 "≈96–102%"。
- **R7.** REVA 的 oracle AUC、SLIC、visual veto 及 COCO/ctx60 转移数字源于早期 stage，本月 RESULTS.md 不含；建议在 artifact 清单中列出对应 run 文件保证可回溯（尤其 M8 的 COCO 基线）。
- **R8.** 审计 "rare-control"→"non-canonical-control"、"equally-rare"→"matched non-canonical"（见第四节残留清单），与 W10 评审整改承诺对齐。

---

**总计：必改 8 项（M1–M8），建议 8 项（R1–R8）。** 除上述条目外，两篇摘要/引言/结论的全部量化主张均能落到具体表格、小节或 RESULTS.md 条目，且 test-300/单种子/style-reimpl/单细胞/in-distribution 限定语总体到位。

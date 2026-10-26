# HRNet-W32 Architecture Mapping & Breakdown

This document provides a detailed breakdown of the provided PyTorch `HighResolutionNet` module hierarchy and illustrates how each component maps directly to the concepts, stages, and mathematical formulations described in *"Deep High-Resolution Representation Learning for Human Pose Estimation"* (Sun et al., CVPR 2019 / TPAMI 2019).

The printed network corresponds to **HRNet-W32** configured with the **progressive downsampling classification head** (HRNet-Cls / HRNetV2p), where the base width is $C = 32$.

---

## 1. High-Level Architecture Flowchart

```
Input Image: (3, H, W)
      │
[ Stem: conv1 + conv2 ] ──> (64, H/4, W/4)
      │
[ Stage 1: layer1 (4 Bottlenecks) ] ──> (256, H/4, W/4)
      │
[ transition1 ] ────────────────┬───────────────────┐
                                │                   │
                        Branch 1 (32, H/4)   Branch 2 (64, H/8)
                                │                   │
                        [ Stage 2: 1 HRModule with 2x2 Multi-Scale Fusion ]
                                │                   │
[ transition2 ] ────────────────┼───────────────────┼───────────────────┐
                                │                   │                   │
                        Branch 1 (32, H/4)   Branch 2 (64, H/8)   Branch 3 (128, H/16)
                                │                   │                   │
                        [ Stage 3: 4 HRModules with 3x3 Multi-Scale Fusion ]
                                │                   │                   │
[ transition3 ] ────────────────┼───────────────────┼───────────────────┼───────────────────┐
                                │                   │                   │                   │
                        Branch 1 (32, H/4)   Branch 2 (64, H/8)   Branch 3 (128, H/16)  Branch 4 (256, H/32)
                                │                   │                   │                   │
                        [ Stage 4: 3 HRModules with 4x4 Multi-Scale Fusion ]
                                │                   │                   │                   │
[ incre_modules ]        (128, H/4)          (256, H/8)          (512, H/16)        (1024, H/32)
                                │                   │                   │                   │
[ downsamp_modules ]            └───── conv3x3 ────>┴───── conv3x3 ────>┴───── conv3x3 ────>│
                                                                                            │
                                                                                    (1024, H/32)
                                                                                            │
                                                                           [ final_layer: 1x1 conv ]
                                                                                            │
                                                                                    (2048, H/32)
                                                                                            │
                                                                                [ global_pool (GAP) ]
                                                                                            │
                                                                             [ classifier (Linear) ]
                                                                                            │
                                                                                    1000 Class Scores
```

---

## 2. Module-by-Module Specification and Paper Mapping

### Module 1: Stem Net
* **PyTorch Components:**
  * `conv1`: `Conv2d(3, 64, kernel_size=3, stride=2, padding=1)` + `bn1` + `act1`
  * `conv2`: `Conv2d(64, 64, kernel_size=3, stride=2, padding=1)` + `bn2` + `act2`
* **Paper Mapping:** Stem network.
* **Function:** Decreases input spatial dimensions by a factor of 4 ($4\times$ downsampling) using two consecutive stride-2 convolutions before entering the parallel subnetwork design.
* **Output Tensor:** $(B, 64, \frac{H}{4}, \frac{W}{4})$

---

### Module 2: Stage 1
* **PyTorch Components:**
  * `layer1`: A `Sequential` block containing 4 `Bottleneck` residual blocks.
  * `layer1[0]` uses `downsample` ($1 \times 1$ conv + BN) to project channels from $64 \rightarrow 256$.
  * `layer1[1..3]` repeat the Bottleneck ($1\times1 \rightarrow 3\times3 \rightarrow 1\times1$) keeping 256 channels.
* **Paper Mapping:** First stage subnetwork ($N_{11}$).
* **Function:** Operates solely on a single high-resolution stream at $\frac{H}{4} \times \frac{W}{4}$. Extracts foundational low-level feature representations.
* **Output Tensor:** 1 branch: $(B, 256, \frac{H}{4}, \frac{W}{4})$

---

### Module 3: First Transition Unit
* **PyTorch Components:**
  * `transition1[0]`: `Conv2d(256, 32, kernel_size=3, stride=1, padding=1)` + BN + ReLU
  * `transition1[1]`: `Conv2d(256, 64, kernel_size=3, stride=2, padding=1)` + BN + ReLU
* **Paper Mapping:** Transition layer spawning 2 parallel resolution streams.
* **Function:**
  * Branch 0 keeps $\frac{H}{4} \times \frac{W}{4}$ and reduces channels to $C = 32$.
  * Branch 1 downsamples spatially to $\frac{H}{8} \times \frac{W}{8}$ via stride 2 and sets channels to $2C = 64$.
* **Output Tensors:** 2 parallel branches:
  * Branch 1: $(B, 32, \frac{H}{4}, \frac{W}{4})$
  * Branch 2: $(B, 64, \frac{H}{8}, \frac{W}{8})$

---

### Module 4: Stage 2
* **PyTorch Components:**
  * `stage2`: Contains 1 `HighResolutionModule`.
  * `branches[0]`: 4 `BasicBlock` units processing Branch 1 ($32 \rightarrow 32$).
  * `branches[1]`: 4 `BasicBlock` units processing Branch 2 ($64 \rightarrow 64$).
  * `fuse_layers`: $2 \times 2$ Exchange Unit matrix:
    * `[0][0]`: `Identity()` (Branch $1 \rightarrow 1$)
    * `[0][1]`: $1 \times 1$ Conv ($64 \rightarrow 32$) + `Upsample(scale=2.0)` (Branch $2 \rightarrow 1$)
    * `[1][0]`: $3 \times 3$ Conv, stride 2 ($32 \rightarrow 64$) (Branch $1 \rightarrow 2$)
    * `[1][1]`: `Identity()` (Branch $2 \rightarrow 2$)
* **Paper Mapping:** 2nd stage parallel subnetwork with multi-scale fusion.
* **Function:** Extracts features in parallel across 2 resolutions and exchanges cross-resolution information once at the end of the stage.
* **Output Tensors:** 2 parallel branches of identical spatial dimensions and channels as input.

---

### Module 5: Second Transition Unit
* **PyTorch Components:**
  * `transition2[0..1]`: 2 $\times$ `Identity()`
  * `transition2[2]`: `Conv2d(64, 128, kernel_size=3, stride=2, padding=1)` + BN + ReLU
* **Paper Mapping:** Transition layer expanding from 2 to 3 branches.
* **Function:** Preserves existing Branches 1 and 2, and downsamples Branch 2 by stride 2 to spawn Branch 3 ($4C = 128$ channels, $\frac{H}{16} \times \frac{W}{16}$).
* **Output Tensors:** 3 parallel branches:
  * Branch 1: $(B, 32, \frac{H}{4}, \frac{W}{4})$
  * Branch 2: $(B, 64, \frac{H}{8}, \frac{W}{8})$
  * Branch 3: $(B, 128, \frac{H}{16}, \frac{W}{16})$

---

### Module 6: Stage 3
* **PyTorch Components:**
  * `stage3`: Sequential list of 4 `HighResolutionModule`s.
  * In each module:
    * `branches[0..2]`: 4 `BasicBlock`s each across channels 32, 64, and 128.
    * `fuse_layers`: $3 \times 3$ full Exchange Unit:
      * **Upsampling ($i > k$):** $1 \times 1$ Conv + nearest upsampling ($2\times$ or $4\times$).
      * **Downsampling ($i < k$):** Contiguous strided $3 \times 3$ convolutions (1 conv for $2\times$, 2 chained convs for $4\times$).
      * **Identity ($i = k$):** Passes representation untouched.
* **Paper Mapping:** 3rd stage subnetwork; 4 repeated multi-scale fusion steps.
* **Function:** Ensures deep cross-talk across all 3 resolutions repeatedly.
* **Output Tensors:** 3 parallel branches: $(32, \frac{H}{4})$, $(64, \frac{H}{8})$, and $(128, \frac{H}{16})$.

---

### Module 7: Third Transition Unit
* **PyTorch Components:**
  * `transition3[0..2]`: 3 $\times$ `Identity()`
  * `transition3[3]`: `Conv2d(128, 256, kernel_size=3, stride=2, padding=1)` + BN + ReLU
* **Paper Mapping:** Transition layer expanding from 3 to 4 branches.
* **Function:** Preserves branches 1, 2, and 3, and downsamples branch 3 to initialize the 4th branch ($8C = 256$ channels, $\frac{H}{32} \times \frac{W}{32}$).
* **Output Tensors:** 4 parallel branches:
  * Branch 1: $(B, 32, \frac{H}{4}, \frac{W}{4})$
  * Branch 2: $(B, 64, \frac{H}{8}, \frac{W}{8})$
  * Branch 3: $(B, 128, \frac{H}{16}, \frac{W}{16})$
  * Branch 4: $(B, 256, \frac{H}{32}, \frac{W}{32})$

---

### Module 8: Stage 4
* **PyTorch Components:**
  * `stage4`: Sequential list of 3 `HighResolutionModule`s.
  * In each module:
    * `branches[0..3]`: 4 `BasicBlock`s per branch for channels 32, 64, 128, and 256.
    * `fuse_layers`: $4 \times 4$ Exchange Unit handling cross-resolution transfer across all 4 scales:
      * Upsampling scale factors up to $8\times$.
      * Downsampling up to 3 consecutive strided $3 \times 3$ convolutions ($8\times$ spatial reduction).
* **Paper Mapping:** 4th stage subnetwork containing 3 exchange modules.
* **Output Tensors:** 4 parallel branches with fused high- and low-resolution representations.

---

### Module 9: Progressive Aggregation Classification Head
* **PyTorch Components:**
  * `incre_modules[0..3]`:
    * Branch 1: Bottleneck ($32 \rightarrow 128$)
    * Branch 2: Bottleneck ($64 \rightarrow 256$)
    * Branch 3: Bottleneck ($128 \rightarrow 512$)
    * Branch 4: Bottleneck ($256 \rightarrow 1024$)
  * `downsamp_modules[0..2]`:
    * `downsamp_modules[0]`: $3 \times 3$ Conv, stride 2 ($128 \rightarrow 256$) added to Branch 2.
    * `downsamp_modules[1]`: $3 \times 3$ Conv, stride 2 ($256 \rightarrow 512$) added to Branch 3.
    * `downsamp_modules[2]`: $3 \times 3$ Conv, stride 2 ($512 \rightarrow 1024$) added to Branch 4.
  * `final_layer`: $1 \times 1$ Conv ($1024 \rightarrow 2048$) + BN + ReLU.
  * `global_pool`: Global average pooling $\rightarrow (B, 2048)$.
  * `classifier`: Linear layer ($2048 \rightarrow 1000$).
* **Paper Mapping:** Section 4 / Appendix classification head (HRNet-Cls / HRNetV2p representation).
* **Function:** Progressively aggregates features from high resolution down to the lowest resolution to form a compact, semantically dense, 2048-dimensional global descriptor.

---

## 3. Summary of Parameters and Branches per Stage

| Stage | Input Branches | Output Branches | Blocks per Branch | Modules in Stage | Channels ($C=32$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stem** | 1 (3 channels) | 1 (64 channels) | 2 Strided Convs | 1 | 64 |
| **Stage 1** | 1 | 1 | 4 Bottlenecks | 1 | 256 |
| **Stage 2** | 2 | 2 | 4 BasicBlocks | 1 | [32, 64] |
| **Stage 3** | 3 | 3 | 4 BasicBlocks | 4 | [32, 64, 128] |
| **Stage 4** | 4 | 4 | 4 BasicBlocks | 3 | [32, 64, 128, 256] |
| **Head** | 4 | 1 | 1 Bottleneck + Conv cascade | 1 | $1024 \rightarrow 2048 \rightarrow 1000$ |
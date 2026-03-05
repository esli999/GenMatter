# Matter-Weighted Evaluation Metrics

## Problem & Motivation

Existing particle-level accuracy metrics weight all particles equally, regardless of their size or importance. This causes severe boundary sensitivity: a visually perfect segmentation can score only ~80% accuracy because tiny boundary particles (10 pixels) with minor errors dominate the metric, despite large central particles (1000+ pixels) being tracked perfectly.

**Goal:** Design an evaluation metric that:
1. Weights particles by their importance (spatial extent × learned probabilistic confidence)
2. Enables fair comparison with point trackers (CoTracker, TAP-Net, TAPIR)
3. Cannot be exploited by adversarial weight configurations
4. Penalizes both false positives and false negatives symmetrically

---

## Metric Definition

### Notation

- $P$ = set of all particles at time $t$
- $M \subseteq P$ = reference particles (object particles identified in frame 0)
- $n_p$ = number of pixels assigned to particle $p$
- $w_p$ = blob weight for particle $p$ (from Dirichlet prior, $\sum_{p \in P} w_p = 1$)
- $f_p \in [0,1]$ = fraction of particle $p$'s pixels inside ground truth mask

### Matter Weight

We define the **matter** associated with particle $p$ as:

$$m_p = n_p \cdot w_p$$

This product captures both spatial extent ($n_p$) and probabilistic importance ($w_p$).

### Evaluation Metrics

**Recall** (what fraction of object matter stays inside the GT mask?):
$$\text{Recall} = \frac{\sum_{p \in M} f_p \cdot m_p}{\sum_{p \in M} m_p}$$

**Precision** (what fraction of predicted matter is actually in the GT mask?):
$$\text{Precision} = \frac{\sum_{p \in M} f_p \cdot m_p}{\sum_{p \in M} f_p \cdot m_p + \sum_{p \in P \setminus M} f_p \cdot m_p}$$

**F1 Score** (harmonic mean, balancing both error types):
$$\text{F1} = \frac{2 \cdot \text{Recall} \cdot \text{Precision}}{\text{Recall} + \text{Precision}}$$

**Interpretation:**
- Numerator of recall: matter from reference particles inside the mask (true positives)
- Denominator of recall: total reference particle matter
- Precision denominator includes background particles entering mask (false positives)

---

## Two Variants: Fixed vs. Adaptive Weights

We compute two versions of each metric, differing in how blob weights are handled:

### Fixed-Weight Metrics (MW-F1-F)

**Definition:** Use blob weights from frame 0 throughout the entire video: $w_p = w_p^{(0)}$ for all frames $t$.

**Rationale:**
- Point trackers (CoTracker, TAP-Net, TAPIR) have no concept of evolving confidence/importance
- For fair comparison, both systems must be evaluated with the **same initial matter distribution**
- Fixed weights cannot be gamed by adaptively downweighting difficult-to-track particles
- This is the primary metric for cross-method comparisons

### Adaptive-Weight Metrics (MW-F1-A)

**Definition:** Use per-frame blob weights: $w_p = w_p^{(t)}$ at each frame $t$.

**Rationale:**
- Shows the advantage of probabilistic matter representation
- Particles that drift or become uncertain can be downweighted by the model
- Demonstrates that the model learns tracking confidence online
- This is a supplementary metric to highlight your system's unique capability

**Diagnostic:** If MW-F1-F - MW-F1-A > 0.1, the model may be gaming the adaptive metric by downweighting difficult particles rather than tracking them well.

---

## Why the Product $m_p = n_p \cdot w_p$?

The product term is **critical** for preventing adversarial exploitation.

### Attack 1: Dirichlet Concentration

**Scenario:** Dirichlet prior concentrates all weight on a single particle: $w_1 = 0.999, w_2 = w_3 = \ldots = 0.001$.

**Without $n_p$ term:**
- 1-pixel particle with $w=0.999$ contributes $0.999$ to the metric (dominates)
- 1000-pixel particle with $w=0.001$ contributes $0.001$ to the metric (negligible)
- **Exploit:** Track only the 1-pixel particle perfectly → high score

**With $n_p$ term:**
- 1-pixel particle: $m = 1 \times 0.999 = 0.999$
- 1000-pixel particle: $m = 1000 \times 0.001 = 1.0$ (actually **more** matter!)
- **Result:** Spatial extent naturally counterbalances weight concentration

### Attack 2: Uniform Weights (Graceful Degradation)

**Scenario:** All particles get equal weight: $w_p = \frac{1}{|P|}$ for all $p$.

**Behavior:**
- Large particles (1000 pixels): $m = 1000 \times \frac{1}{500} = 2.0$
- Small particles (10 pixels): $m = 10 \times \frac{1}{500} = 0.02$
- **Result:** Metric reduces to pixel-count weighting, which is still a valid importance measure

The product term ensures that neither spatial extent nor weight can be ignored—both must contribute to matter.

---

## Comparing with Point Trackers

To fairly evaluate GenParticles against point trackers (CoTracker, TAP-Net, TAPIR):

### Protocol

1. **Initialize both systems identically:**
   - Run GenParticles frame 0 initialization (K-means + initial Gibbs sweeps)
   - Extract $\{(\mu_p^{(0)}, w_p^{(0)}) : p \in M\}$ for reference particles
   - Initialize point tracker with these same positions $\mu_p^{(0)}$

2. **Track through video:**
   - GenParticles: Run full probabilistic tracking with Kalman + HDGMM
   - Point tracker: Track points $\{x_p^{(t)}\}$ using their method

3. **Evaluate with identical weighting:**
   - For GenParticles: Use $m_p = n_p^{(t)} \cdot w_p^{(0)}$ (fixed frame-0 weights)
   - For point tracker: Use $m_p = 1 \cdot w_p^{(0)}$ (each point is 1 pixel, use same frame-0 weights)
   - Compute MW-F1-F for both systems using same formulas

4. **Report:** Both systems evaluated on same initial distribution with same weighting scheme.

**Key insight:** Point trackers naturally have $n_p = 1$ (one pixel per point). The fixed-weight protocol ensures fair comparison despite different representations.

---

## Diagnostic Checks for Metric Gaming

Monitor these four indicators to detect potential adversarial behavior:

| Check | Formula | Expected Range | Warning Threshold | Interpretation |
|-------|---------|----------------|-------------------|----------------|
| **Adaptive-Fixed Gap** | MW-F1-F - MW-F1-A | $[-0.05, 0.05]$ | > 0.1 | Large positive gap suggests model is gaming by downweighting difficult particles in the adaptive variant |
| **Weight Entropy** | $H = -\sum w_p \log w_p / \log \|P\|$ | $[0.5, 1.0]$ | < 0.5 | Low entropy indicates weights are concentrated on few particles (Dirichlet concentration attack) |
| **Spatial Std Dev** | $\sigma = \text{std}(\mu_p : p \in M)$ | > 0.5 | < 0.1 | Low spatial spread indicates particles are clustered and not covering the object's full extent |
| **Ref Particles** | $\|M\|$ | ≥ 10 | < 10 | Too few reference particles makes the metric unstable and sensitive to individual particle errors |

**Normalized entropy:** $H = 1.0$ means uniform distribution (all weights equal), $H = 0$ means all weight on one particle.

---

## Alternatives Considered & Rejected

### Per-Particle IoU

**Definition:** $\text{IoU}_p = \frac{n_p f_p}{n_p + |\text{GT}| - n_p f_p}$

**Why rejected:** The union denominator is dominated by the full GT mask size ($|\text{GT}| \sim 50\text{k}-200\text{k}$ pixels), while individual particles have only $n_p \sim 100-2000$ pixels. This makes per-particle IoU values < 0.01, which are numerically meaningless.

### Coverage-Only (Unbalanced Recall)

**Definition:** $\frac{\sum_{p \in M} f_p \cdot m_p}{\sum_{p \in M} m_p}$ (recall without precision)

**Why rejected:** This metric only measures what fraction of object matter stays inside the mask, but completely ignores false positives (background particles incorrectly entering the mask). It's asymmetric and can be trivially gamed: put all weight on one tiny particle inside the mask → 100% score.

**Key insight:** We need both recall (object matter staying in) **and** precision (background matter staying out) for a balanced evaluation.

---

## Relationship to Unweighted Metrics

### Unweighted FN/FP Rates

These are the original discrete particle counts:
- $\text{FN Rate} = \frac{|\{p \in M : f_p < 0.5\}|}{|M|}$ (fraction of object particles drifting out)
- $\text{FP Rate} = \frac{|\{p \in P \setminus M : f_p > 0.5\}|}{|P \setminus M|}$ (fraction of background particles entering)

**Limitations:** Treats all particles equally regardless of size or importance.

**When to use:** Understanding discrete particle-level behavior, visualizations, debugging.

### Matter-Weighted Metrics (This Work)

Weight each particle by $m_p = n_p \cdot w_p$ to emphasize important particles.

**Advantages:**
- Spatially large particles contribute more than tiny boundary particles
- Uses learned probabilistic weights to capture model confidence
- Balanced (penalizes both FP and FN)
- Enables fair comparison with point trackers

**When to use:** Primary evaluation metric for papers, benchmarks, cross-method comparisons.

---

## Implementation Details

### Code Location

- **Core implementation:** `genparticles/evaluation.py` lines 319-442
- **Module documentation:** `genparticles/evaluation.py` lines 1-38
- **Diagnostic checks:** `dino_tracking.py` lines 997-1043
- **JSON output:** All metrics saved to `experiment_results.json`

### Usage Recommendations

1. **Always report MW-F1-F for comparisons** with other methods (immune to adaptive gaming)
2. **Report MW-F1-A as supplementary** to show your system's advantage
3. **Report all three metrics** (recall, precision, F1) for full error type breakdown
4. **Monitor diagnostics** to ensure metrics aren't being gamed
5. **Include unweighted FN/FP rates** as complementary discrete-level view

---

## Summary

**Problem:** Unweighted particle metrics are boundary-sensitive and don't reflect true tracking quality.

**Solution:** Matter-weighted recall/precision/F1 using $m_p = n_p \cdot w_p$.

**Key decisions:**
1. Product term prevents adversarial gaming (neither $n_p$ nor $w_p$ alone is sufficient)
2. Two variants (fixed/adaptive) enable fair comparison while highlighting probabilistic benefit
3. Balanced metrics (both recall and precision) prevent exploitation
4. Diagnostic checks detect potential gaming behaviors

**Result:** Robust, interpretable evaluation that fairly compares particle tracking with point tracking while emphasizing spatially and probabilistically important particles.

# Final Project Report

Best Experiment: Exp_SoftCE_Random_Linear
- KL Divergence: 1.1590893
- Calibration ECE: 0.0685848718583583

### Robustness Integration
- FGSM High Entropy Acc Drop: 0.2200
- PGD High Entropy Acc Drop: 0.2600
- FGSM Low Entropy Acc Drop: 0.5800
- PGD Low Entropy Acc Drop: 0.6400

### Grad-CAM Analysis
The low entropy images, including a bird in flight against open sky, a horse standing in profile, a truck with clear side markings, a red dump truck, and a gold car, all show unambiguous subjects with strong contrast against simple backgrounds. Their Grad-CAM heatmaps concentrate on tight, localized regions because the model can lock onto one decisive visual cue, such as the bird's wing silhouette or the truck's body shape. The annotator distributions match this behavior: each image is dominated by a single class with probability near 1.0, indicating near-unanimous agreement among the 50 annotators.

The high entropy images show the opposite pattern. Model attention is diffuse or split across multiple regions, and the annotator distributions explain why. In these cases the images are visually ambiguous, low quality, oddly posed, partially occluded, or genuinely difficult even for humans at 32x32 resolution. The uncertainty in the heatmaps reflects real ambiguity in the underlying labels rather than a purely model-specific failure.

### Manual Disagreement Source Analysis
- High entropy #1 (idx 1754, H=2.21): A cat or dog in low light against a dark background. The label distribution is spread across bird (class 2), cat (3), dog (5), horse (7), and truck (9). Likely cause: poor image quality plus ambiguous animal identity. The dark background obscures body shape, making the animal difficult to identify even for humans.
- High entropy #2 (idx 1120, H=2.19): What appears to be a cat or dog lying down near a doorway from an unusual angle. The distribution is spread across cat, deer, dog, and horse. Likely cause: unusual viewpoint plus a boundary case between classes. The prone pose removes many of the body shape cues that annotators normally rely on.
- High entropy #3 (idx 1995, H=2.16): A small animal, possibly a frog or bird, being held in a hand against green foliage. The distribution is spread across bird, cat, deer, dog, and frog. Likely cause: ambiguous object identity plus occlusion. The hand partially blocks the animal, and its small size makes species identification difficult at 32x32 resolution.
- High entropy #4 (idx 1034, H=2.10): A dark scene with what looks like a deer or other large animal against vegetation and a blue stripe. The distribution is dominated by deer (class 4) but still places meaningful mass on cat and horse. Likely cause: poor image quality plus distracting scene content. The blue element introduces noise, and the animal pose is ambiguous enough to blur the deer-versus-horse distinction.
- High entropy #5 (idx 1105, H=2.05): A grayscale or desaturated image of what appears to be a truck or automobile with machinery. The distribution is split between airplane (0), automobile (1), ship (8), and truck (9). Likely cause: genuine multi-label style ambiguity plus missing color information. Desaturation removes an important discriminative cue, and the object itself may be a specialized vehicle or machinery that does not cleanly match a single CIFAR-10 class.

### Failure Case Analysis
The strongest failure case is high entropy #5. The target distribution assigns meaningful probability to airplane, automobile, ship, and truck even though those classes are semantically far apart. This is difficult for any model, not because of a simple modeling mistake, but because the visual evidence itself is ambiguous and color cues are largely absent. A model trained on soft labels should spread probability mass across several plausible classes, but the exact split remains hard to calibrate consistently.

High entropy #1 is the second notable failure case. The image is so dark that the class is effectively underdetermined by the pixel content alone at 32x32 resolution. In this case, model uncertainty reflects information loss in the input rather than a straightforward recognition error.

### Architecture Rationale
#### Why the Linear Head Was Preferred Over the MLP Head
The linear head was chosen as the primary architecture because the task is distribution prediction rather than complex nonlinear classification. A single linear layer followed by softmax maps the 512-dimensional ResNet-18 feature vector directly to a 10-dimensional probability distribution, which keeps the output interpretation simple: each logit is a weighted sum of learned backbone features. The MLP head adds a hidden layer and dropout, increasing capacity in a setting with only 6,000 soft-label training images and inherently noisy targets aggregated from roughly 50 annotators per sample. That extra capacity increases the risk of overfitting to idiosyncratic patterns in the training distributions. The experimental results support the simpler choice: Exp_KL_Random_Linear achieved competitive KL divergence, indicating that the backbone features were already expressive enough for the task without requiring a more complex prediction head.

#### Why Random Initialization Was Used, and Why Hard-Label Pretraining Was Avoided
The 50,000 CIFAR-10 hard-label training images were intentionally not used in this project. The core reason is asymmetry in the supervision signal: hard labels collapse the full annotator distribution into a single majority-vote class, discarding exactly the disagreement structure this project aims to model. Pretraining on hard labels would bias the backbone toward confident single-class discrimination, which works against the goal of calibrated uncertainty prediction. Random initialization forces the network to learn feature representations directly from the soft-label objective from the first update onward, so the gradients always reflect distribution matching rather than hard classification. The tradeoff is slower convergence and potentially weaker low-level features than ImageNet or hard-label pretraining might provide, but given the small 32x32 input resolution and the close domain match within CIFAR-10, random initialization proved sufficient in practice.

### Loss Function Justification
#### KL Divergence
KL divergence measures how much information is lost when the predicted distribution q is used to approximate the true annotator distribution p: KL(p || q) = sum p(y) log(p(y) / q(y)). It is a natural loss for this task because it directly penalizes the model for assigning too little probability to classes that human annotators frequently selected. Its asymmetry is useful here: underestimating high-probability classes is treated as a more serious error than slightly overestimating low-probability classes. The results show that the KL-based model achieved the best expected calibration error among the compared losses, which supports its suitability for uncertainty-aware distribution prediction.

#### Jensen-Shannon Divergence
Jensen-Shannon divergence is a symmetric and bounded variant of KL divergence: JSD(p || q) = 0.5 * KL(p || M) + 0.5 * KL(q || M), where M = 0.5 * (p + q). Because it is symmetric, it penalizes underestimation and overestimation more evenly across classes. In theory this gives a more balanced objective, but in this project it performed slightly worse on KL divergence and entropy-correlation style metrics. That outcome suggests the asymmetry of KL was actually beneficial, because matching the dominant human choices mattered more than enforcing fully symmetric penalties across the whole distribution.

#### Soft Cross-Entropy
Soft cross-entropy is CE(p, q) = -sum p(y) log q(y), which differs from KL divergence only by the entropy of the target distribution H(p). Since H(p) is constant with respect to the model parameters, minimizing soft cross-entropy is mathematically equivalent to minimizing KL divergence during training. Despite that theoretical equivalence, Exp_SoftCE_Random_Linear achieved the best KL divergence on the test set, likely because of numerical differences in optimization behavior and gradient scaling. This makes soft cross-entropy a practical and fully justified objective for the same reason as KL divergence: it rewards accurate matching of the full annotator distribution rather than only the top class.

#### Custom Disagreement Loss
The custom disagreement loss was designed to capture something the standard distribution-matching losses do not explicitly enforce: entropy matching. The loss is defined as KL(p || q) + alpha * MSE(H(p), H(q)), where H is Shannon entropy and alpha = 0.5. The motivation is that KL divergence alone does not directly penalize errors in the overall level of uncertainty. A model can have a reasonable KL value while still systematically underestimating or overestimating how spread out the distribution should be. The added entropy-MSE term encourages the model to match not only the class probabilities but also the total disagreement magnitude. In practice, this loss produced competitive accuracy but a higher calibration error than KL alone, suggesting that explicitly enforcing entropy alignment introduced a modest calibration tradeoff.

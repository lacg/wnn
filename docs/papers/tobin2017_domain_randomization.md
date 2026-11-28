# Tobin, Fong, Ray, Schneider, Zaremba, Abbeel 2017 — Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World

IROS 2017. arXiv:1703.06907. https://arxiv.org/abs/1703.06907 — Local PDF: `papers/tobin2017_domain_randomization.pdf`. Read 28/11/2026.

**Relevance.** Origin of the "domain randomization" term — but VISUAL only. **Regime:** object localization from
camera images (MuJoCo renderer), Fetch arm grasping; real images = test.

## Verbatim extraction
- Randomized "for each" training sample: "Number and shape of distractor objects on the table"; "Position and texture
  of all objects on the table"; "Textures of the table, floor, skybox, and robot"; camera position/orientation/FOV;
  "Number of lights in the scene"; "Type and amount of random noise added to images". Camera orientation "offset by up
  to 0.1 radians". "The textures of all objects are chosen uniformly at random".
- Ablation (Sec. C, main text, Fig. 5): sensitivity to number of training images, number of unique textures, random
  noise in pre-processing, distractors, camera-position randomization; "at least somewhat sensitive to all of the
  factors except the use of random noise."

## Absent
- NO physical/dynamics parameters randomized (no mass, inertia, actuator, latency). Not a control-robustness protocol;
  cite only for the DR concept.

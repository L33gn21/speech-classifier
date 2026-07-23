# PRD — Speech Classifier (Accent Classification + AI Synthetic Speech Detection)

- Status: Draft
- Last updated: 2026-07-23

---

## 1. Overview

A multitask speech classification service that takes an English audio clip and
**(1) estimates the speaker's accent (nationality) as per-class probabilities**, and
**(2) determines whether the clip is a real human recording (REAL) or an
AI-synthesized voice (FAKE)**.

### 1.1 Background / Problem

- Accent is a recurring, useful signal for language-learning feedback,
  call-center/customer-service quality monitoring, and speaker segmentation, but
  manual human judgment is expensive and subjective.
- As generative AI voices (TTS, voice cloning) become mainstream, voice-driven
  services increasingly need to verify that an input is genuinely human (e.g., to
  counter voice phishing and malicious deepfake audio).
- Both problems share the same shape — a classification head on top of a speech
  encoder — so we unify them into one multitask model that shares a backbone,
  cutting both development and serving cost.

### 1.2 Goals

1. Produce reliable per-class accent probabilities (`US/UK/CA/AU/IN/CN`) for English
   speech input.
2. Jointly determine REAL/FAKE (synthetic-speech) status for the same input.
3. Generalize well enough that performance doesn't collapse on unseen corpora or
   unseen spoof attack types.
4. Ship the model as a web demo (upload → view results) so anyone can verify it.

### 1.3 Non-Goals

- Pronunciation coaching (phoneme-alignment-based GOP) is out of scope for this
  phase (roadmap Level 3, separate initiative).
- A Korean (KR) accent class is out of scope for this phase (data export
  restrictions require a separate track).
- Real-time streaming inference is not a goal — the target is batch/single-shot
  inference on uploaded clips.
- Speaker identification (who is speaking) is not addressed — only accent-region
  estimation and authenticity judgment are in scope.

---

## 2. Users and Use Cases

| User | Need | Scenario |
|------|------|----------|
| Language/education service operator | Understand a learner's accent tendencies | Upload a recording → view accent percentages |
| Call-center / customer-support QA manager | Estimate caller background | Upload a call clip → view accent breakdown |
| Security / trust & safety owner | Verify inbound audio is genuinely human | Upload suspect audio → view REAL/FAKE verdict |
| Researcher / demo user | Get a feel for model performance | Upload a clip on the web demo and view results |

---

## 3. Functional Requirements

### 3.1 Accent (country) classification

- Input: an English speech clip from a human speaker (audio file).
- Output: per-class probabilities (percentages) — `US, UK, CA, AU, IN, CN`.
- Frame-level logits must be preserved so accent strength can be visualized as a
  time-axis heatmap.

### 3.2 AI synthetic speech (FAKE) detection

- Input: the same audio clip as in 3.1 (shared backbone).
- Output: REAL/FAKE binary probability.
- Samples without a country label (e.g., non-English or unknown origin) must still
  be usable for training via the fake label alone (country label handled with an
  `ignore_index`).

### 3.3 Inference / demo

- A web demo must accept a single file upload and return accent percentages, a
  REAL/FAKE banner, and (where available) a frame-level heatmap.
- The demo must let users compare multiple trained model versions via a dropdown.

### 3.4 Training / evaluation pipeline

- The full path — data curation (`curated/<CC>/manifest.csv`) → speaker-disjoint
  train/val/test split → training → holdout evaluation — must be scripted and
  reproducible.
- Country-only training and multitask (country+fake) training must be switchable
  from a single training script via a flag (`--multitask`).
- Metrics on unseen corpora (e.g., VoxForge) and unseen spoof attacks (ASVspoof
  attacks not seen in training) are the primary basis for model selection, to avoid
  optimistic bias from the validation set.

---

## 4. Non-Functional Requirements

- **Reproducibility**: every training experiment records its hyperparameters, data
  snapshot, and results in a report (`reports/`).
- **Cost control**: before submitting a GPU training job, estimated time and cost
  must be calculated and reported.
- **Data governance**: raw data is managed only in cloud storage, never kept in bulk
  locally. Per-country data export restrictions (e.g., Korean data may not leave
  Korea) must be respected.
- **Generalization**: performance is tracked not only on training corpora but also
  on unseen corpora / unseen attack types, with degradation quantified.

---

## 5. Success Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| Country macro-F1 (unseen corpus) | Macro-F1 across the 6 accent classes | Tracked continuously; investigate immediately on regression |
| Fake macro-F1 (unseen attacks) | REAL/FAKE binary macro-F1 | Target 0.90+ |
| Multitask macro-F1 | mean(country macro-F1, fake macro-F1) | Target 0.70+ |
| Demo availability | Success rate of upload → result round trip | All deployed models infer successfully |

---

## 6. Milestones / Roadmap

- **Level 1 (done)**: utterance-level accent percentage estimation.
- **Level 1.5 (done)**: accent + synthetic-speech multitask model, web demo deployed.
- **Level 2 (available)**: further develop the time-axis accent heatmap using
  frame-level logits.
- **Level 3 (separate initiative)**: phoneme-alignment-based pronunciation coaching
  (GOP family) — requires a read-speech assumption and a reference definition first.
- **KR track**: add a Korean accent class using AI-Hub data (run in a separate
  region due to data export restrictions).

---

## 7. Risks and Open Questions

- **North American (US/CA) accent confusion**: the largest remaining risk is
  US/CA misclassification on unseen corpora. Domain randomization augmentation has
  mitigated it, but a root fix (diversifying CA training data) is still in progress.
- **Minority-class (CN) recall**: CN has the least training data and lower recall
  than other classes. More data is needed.
- **Multitask loss weighting (λ)**: the balance between country/fake loss weights
  has flipped the ranking between validation and unseen-attack test sets in past
  experiments — model selection must always be reconfirmed on unseen data.
- **Evaluation script stability**: the multitask scoring path in the `evaluate.py`
  CLI hangs under certain conditions (low priority); the training job's built-in
  scoring is used as a workaround in the meantime.

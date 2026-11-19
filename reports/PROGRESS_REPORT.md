# Progress report — Milestone 2 checkpoint

Team: **[ENTER TEAM NAME]**  
Members: **[ENTER MEMBER NAMES]**  
Course/section: **[ENTER COURSE AND SECTION]**

## Weighted project status

| Full-project task | Weight | Owner | Checkpoint status | Earned progress | Evidence |
| --- | ---: | --- | --- | ---: | --- |
| Requirements and interface design | 10% | Team | Complete | 10% | Responsive dashboard and controls |
| Core browser camera and inference | 20% | Team | Complete | 20% | `index.html`, `src/js/app.js`, vendored models |
| Dataset acquisition, cleaning, and split | 15% | Team | Complete | 15% | Data card, manifest, audit, preparation script |
| Baseline training and evaluation | 10% | Team | Half complete | 5% | Reproducible 30-image P/R baseline and first custom-model metrics |
| Phone-domain dataset and retraining | 15% | Team | Not started | 0% | Planned next milestone |
| Accuracy hardening and regression tests | 15% | Team | Planned next milestone | 0% | Failure cases identified |
| Representative-device performance work | 5% | Team | Not started | 0% | Planned Android/iOS benchmarks |
| Final locked evaluation and report | 10% | Team | Not started | 0% | Reserved for final milestone |
| **Total** | **100%** |  |  | **50%** |  |

The 50% value measures delivery against the full project plan; it is not an
accuracy percentage.

## Current blockers

| Blocker | Effect | Planned response |
| --- | --- | --- |
| Public CCTV data differs from phone-camera views | Misses and false positives | Collect consented phone frames and hard negatives |
| Low light and very small people | Reduced recall and unstable boxes | Add representative samples and compare input sizes |
| Mobile startup and compute limits | Slow loading on some phones | Measure asset loading, quantize, and benchmark devices |
| Partial-body and broad group detections | Duplicate or merged counts | Add regression images and tune pose/geometry fusion on validation only |

## Required personalization

Before submission, replace `Team` in the Owner column with the actual member
responsible for each area. Every member should be able to explain at least one
file they own.


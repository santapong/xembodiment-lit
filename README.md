# X-Embodiment Literature Workspace

Research notes and verified references on embodied AI with the **Open X-Embodiment dataset** (arXiv:2310.08864) and the vision-language-action (VLA) model lineage it enabled.

## Contents

- `paper/references.bib` — 10 BibTeX entries, all fetched programmatically from arXiv (RT-X, Octo, OpenVLA, Diffusion Policy, RT-1/RT-2, π0, DROID, GR00T N1, CogACT)
- `notes/survey.md` — field survey: OXE foundation → open generalist policies → newer wave (2024–2026), with key numbers and themes
- `notes/openvla-deep-dive.md` — OpenVLA architecture, training data, results, efficiency methods, limitations

## Quick Facts

| Model | Params | Data | Headline result |
|---|---|---|---|
| RT-2-X | 55B | OXE mixture | ~3× generalization vs single-embodiment |
| Octo | 27M–93M | 800k OXE trajs | First fully open generalist policy |
| OpenVLA | 7B | 970k OXE trajs | Beats closed RT-2-X by 16.5% absolute |
| π0 | ~3B VLM + action expert | 7 platforms | Flow matching @ 50Hz, dexterous SOTA |
| GR00T N1 | 2.2B | real+video+synthetic | Open humanoid foundation model |

## Key Resources

- Dataset: https://robotics-transformer-x.github.io
- Code: https://github.com/google-deepmind/open_x_embodiment

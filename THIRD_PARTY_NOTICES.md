# Third-party notices

The root MIT license covers the RAFC integration and downstream task code in
this repository. Pretrained backbones and their checkpoints remain subject to
their respective upstream terms.

- LWM1.1 integration targets [wi-lab/lwm-v1.1](https://huggingface.co/wi-lab/lwm-v1.1).
  As checked on 24 August 2026, the upstream repository declares no license or
  explicit grant of use, modification, or redistribution rights. The official
  pretrained checkpoint is therefore not redistributed. Obtain permission and
  the checkpoint directly from the upstream author before use.
- ContraWiMAE is derived from
  [WirelessContrastiveMaskedLearning](https://github.com/BerkIGuler/WirelessContrastiveMaskedLearning)
  and retains its MIT license in `models/contrawimae/LICENSE`.
- DeepMIMO v4 is an external Apache-2.0 dependency pinned in
  `requirements-data.txt`. This repository contains only the project-specific
  temporal data recipe and receiver-selection metadata; it does not vendor the
  DeepMIMO package, scenarios, or generated CSI. Scenario/data terms are
  maintained by the [DeepMIMO project](https://www.deepmimo.net/).

# Third-party notices

This project's own code and documentation are under the [MIT License](LICENSE). The MIT License
does **not** cover `ml-depth-pro-2/`, the model weights, nuScenes data, or the nuScenes-derived
figures in `explination/assets/`. Those keep the licenses listed below.

This project combines code, a model and a dataset from other parties. Each keeps its own license.
**Only the items marked "included" are in this repository.** Everything else is downloaded by the
user and is not redistributed here.

## Model and data

| Component | Used for | License | In this repo? | What you must do |
|---|---|---|---|---|
| **Depth Pro 2 code** (Apple), [`ml-depth-pro-2/`](ml-depth-pro-2/) | Model code | Apple license, see [`ml-depth-pro-2/LICENSE`](ml-depth-pro-2/LICENSE) (permits use, modification, redistribution) | **Included, unmodified** | Keep Apple's `LICENSE` notice. Don't use Apple's name or marks to endorse this project. Its own third-party parts are listed in [`ml-depth-pro-2/ACKNOWLEDGEMENTS`](ml-depth-pro-2/ACKNOWLEDGEMENTS) |
| **Depth Pro 2 weights** (Apple) | Model parameters | [Apple Machine Learning Research Model License](ml-depth-pro-2/LICENSE_MODEL): **research use only**, no commercial use | No. Downloaded at run time (`download_data.py --weights`) | Research use only. If you redistribute the weights or a modified/fine-tuned model, include the license and the attribution notice it specifies |
| **nuScenes v1.0-mini** (Motional) | Camera images, LiDAR, calibration | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) **plus** the [nuScenes Terms of Use](https://www.nuscenes.org/terms-of-use) (non-commercial) | No. Downloaded by the user (`download_data.py --nuscenes-mini`) | Non-commercial use. Attribute nuScenes and cite the paper. Share adaptations under the same license. Don't imply Motional endorses your work. Per the Terms of Use, derived data such as tables and charts is also covered |
| **Figures derived from nuScenes**: [`explination/assets/video_frame.jpg`](explination/assets/video_frame.jpg), [`explination/assets/lidar_topdown.png`](explination/assets/lidar_topdown.png) | Illustrations in the write-up | CC BY-NC-SA 4.0 + nuScenes Terms of Use (inherited) | **Included** | Attribution below. Shared under the same license, non-commercial |

**Attribution for the included figures:** adapted from the nuScenes dataset (Motional AD Inc.),
<https://www.nuscenes.org>, licensed under CC BY-NC-SA 4.0. Changes: Depth Pro 2 depth predictions
rendered next to a camera frame (`video_frame.jpg`), and a top-down plot of one LiDAR sweep with the
camera's field of view (`lidar_topdown.png`). Motional does not endorse this project.

## Python packages (installed with pip, not redistributed)

| Package | License |
|---|---|
| PyTorch, torchvision | BSD-3-Clause (PyTorch also bundles components under Apache-2.0, MIT, Zlib and others) |
| rerun-sdk (incl. the Rerun viewer) | MIT OR Apache-2.0 |
| PyAV | BSD-3-Clause. **Note:** the binary wheel bundles FFmpeg plus codec libraries, including x264/x265, which are GPL. We only use them locally to encode MP4s. Redistributing those binaries would carry GPL obligations |
| NumPy | BSD-3-Clause |
| Pillow | MIT-CMU (HPND) |
| pillow-heif | BSD-3-Clause (bundles libheif, LGPL) |
| Matplotlib | Matplotlib license (PSF-based) |
| timm | Apache-2.0 |
| huggingface-hub, safetensors | Apache-2.0 |
| click | BSD-3-Clause |

## Citations

If you use this work, please cite the model and the dataset:

```bibtex
@article{DepthPro2:2026:arxiv,
  title   = {Depth Pro 2: Sharp and Temporally Consistent Metric Monocular Depth},
  author  = {Peiyun Hu and Marko Mihajlovic and Lyndon Duong and Lahav Lipson and Wei Dong and
             Ama\"{e}l Delaunoy and Stephan R. Richter and Vladlen Koltun},
  journal = {arXiv preprint arXiv:2611.26603},
  year    = {2026},
}

@article{nuscenes2019,
  title   = {nuScenes: A multimodal dataset for autonomous driving},
  author  = {Holger Caesar and Varun Bankiti and Alex H. Lang and Sourabh Vora and
             Venice Erin Liong and Qiang Xu and Anush Krishnan and Yu Pan and
             Giancarlo Baldan and Oscar Beijbom},
  journal = {arXiv preprint arXiv:1903.11027},
  year    = {2019},
}
```

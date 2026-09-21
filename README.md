# FetchMan

*Learning Visual Humanoid Loco-Manipulation Policies from Simulated Experiences*

[Website](https://www.orayyan.com/fetchman) · [Paper](https://arxiv.org/abs/2608.17027)

<p align="center">
  <img src="https://github.com/user-attachments/assets/1721c6c0-060a-4661-b4fb-bb7ec2c10746" width="100%" alt="Scripted pick demonstrations collected in parallel across randomized houses">
</p>

- [x] Data Generation
- [ ] HF Data Release
- [ ] Behavior Cloning
- [ ] Reinforcement Learning
- [ ] Benchmark

## Setup

```bash
conda create -n g1 python=3.10 -y && conda activate g1
pip install -r requirements.txt
```

## Usage

### Data Generation

```bash
python main.py --env=molmospaces/configs/example.py
```

`example.py` sets the house, the target object, and the domain randomization
(textures, lighting, placement, height, cameras).

Add `--render` to watch, `--record` to save a LeRobot dataset.

## Citation

```bibtex
@article{rayyan2026fetchman,
  title   = {FetchMan: Learning Visual Humanoid Loco-Manipulation Policies from Simulated Experiences},
  author  = {Rayyan, Omar and Li, Zhi and Argus, Max and Jiang, Yuxin and Yu, Chang and Jiang, Chenfanfu and Cui, Yuchen},
  journal = {arXiv preprint arXiv:2608.17027},
  year    = {2026}
}
```

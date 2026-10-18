# FetchMan

*Learning Visual Humanoid Loco-Manipulation Policies from Simulated Experiences*

[Website](https://www.orayyan.com/fetchman) · [Paper](https://arxiv.org/abs/2608.17027) · [Data](https://huggingface.co/datasets/allenai/fetchman-data)

<p align="center">
  <img src="https://github.com/user-attachments/assets/1721c6c0-060a-4661-b4fb-bb7ec2c10746" width="100%" alt="Scripted pick demonstrations collected in parallel across randomized houses">
</p>

- [x] Data Generation
- [x] HF Data Release
- [x] Behavior Cloning
- [ ] Reinforcement Learning
- [ ] Benchmark

## Setup

```bash
conda create -n g1 python=3.10 -y && conda activate g1
pip install -r requirements.txt -r train/requirements.txt
```

## Example

1. Collect 1000 bowl pick episodes in one house, with 8 workers.

```bash
MAX_EPISODES=1000 bash molmospaces/scripts/collect_parallel.sh 8 0 bowl_pick
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/2c10c28d-acb7-44de-942e-869750cb1f7a" width="100%" alt="Head-camera view of eight scripted bowl picks">
</p>

2. Fold the per-worker shards into one dataset.

```bash
python molmospaces/scripts/merge_datasets.py \
    --input_glob='data/data_*/*' \
    --output_root=data/local/bowl_pick --output_repo_id=local/bowl_pick
```

3. Train on it, with a held-out val split and rollouts in the simulator.

```bash
python -u train/main.py \
    --data_root=data/local/bowl_pick --repo_id=local/bowl_pick \
    --env=molmospaces/configs/bowl_pick.py \
    --devices=0 --agent.batch_size=128 \
    --val_episodes=50 --val_interval=1 \
    --eval_interval=1000 --eval_episodes=25 --eval_num_envs=8 --eval_render_gpu=1
```

<img src="https://github.com/user-attachments/assets/0eb8fa9b-03a4-42c1-841d-4227e88e3006" width="50%" alt="Training loss and evaluation success rate">

## Citation

```bibtex
@article{rayyan2026fetchman,
  title   = {FetchMan: Learning Visual Humanoid Loco-Manipulation Policies from Simulated Experiences},
  author  = {Rayyan, Omar and Li, Zhi and Argus, Max and Jiang, Yuxin and Yu, Chang and Jiang, Chenfanfu and Cui, Yuchen},
  journal = {arXiv preprint arXiv:2608.17027},
  year    = {2026}
}
```

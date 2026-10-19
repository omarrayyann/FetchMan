# FetchMan

*Learning Visual Humanoid Loco-Manipulation Policies from Simulated Experiences*

[Website](https://www.orayyan.com/fetchman) · [Paper](https://arxiv.org/abs/2608.17027) · [Data](https://huggingface.co/datasets/allenai/fetchman-data)

<p align="center">
  <img src="https://github.com/user-attachments/assets/1721c6c0-060a-4661-b4fb-bb7ec2c10746" width="100%" alt="Scripted pick demonstrations collected in parallel across randomized houses">
</p>

- [x] Data Generation
- [x] HF Data Release
- [x] Behavior Cloning
- [x] Reinforcement Learning
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

<img src="https://github.com/user-attachments/assets/327b6cbe-21a7-4fc8-b946-b1d2c5796417" width="100%" alt="Training loss and evaluation success rate">

4. Fine-tune the behavior-cloned policy with RL, scored on a fixed benchmark.

```bash
# export 200 successful resets to replay as a fixed eval set
python main.py --env=molmospaces/configs/bowl_pick.py --episodes=450 \
    --save_json=molmospaces/assets/eval_sets/bowl_pick_200.jsonl

python -u rl/main.py \
    --checkpoint=exp/g1/Debug/<run>/step=00016000.ckpt \
    --env=molmospaces/configs/bowl_pick.py \
    --num_groups=32 --group_size=8 --parallel_envs=64 \
    --noise_level=0.7 --num_sampling_steps=10 --actor_lr=5e-6 \
    --time_limit=40 --max_steps_per_episode=15000 \
    --gpu=0 --render_gpus=1,2,3,4,5,6,7 \
    --eval_every_n_updates=10 --save_every_n_updates=10 \
    --eval_episodes_json=molmospaces/assets/eval_sets/bowl_pick_200.jsonl \
    --run_name=bowl_pick_grpo
```

<img src="https://github.com/user-attachments/assets/2b4d1b00-ec03-4f3a-952f-8c6531dabc6d" width="100%" alt="Benchmark success rate, episode length, and fraction of groups giving gradient">

## Citation

```bibtex
@article{rayyan2026fetchman,
  title   = {FetchMan: Learning Visual Humanoid Loco-Manipulation Policies from Simulated Experiences},
  author  = {Rayyan, Omar and Li, Zhi and Argus, Max and Jiang, Yuxin and Yu, Chang and Jiang, Chenfanfu and Cui, Yuchen},
  journal = {arXiv preprint arXiv:2608.17027},
  year    = {2026}
}
```

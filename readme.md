# HOPE: A Reinforcement Learning-based Hybrid Policy Path Planner for Diverse Parking Scenarios
![pipeline](assets/algo_struct.png)

```mermaid
graph TD
    subgraph MultiObsEmbedding
        input[input]

        subgraph Embedding_Layers
            lidar_input[lidar: b, l]
            target_input[target: b, t]
            action_mask_input[action_mask: b, a]
            img_input[img: b, c, w, h]
            action_input[action: b, act]

            embed_lidar[Embed Lidar]
            embed_tgt[Embed Target]
            embed_am[Embed Action Mask]
            embed_img[Embed Image]
            re_embed_img[Re-Embed Image]
            embed_action[Embed Action]

            lidar_input --> embed_lidar
            target_input --> embed_tgt
            action_mask_input --> embed_am
            img_input --> embed_img
            embed_img --> re_embed_img
            action_input --> embed_action

            embed_lidar --> feature_lidar[feature_lidar: b, embed_size]
            embed_tgt --> feature_target[feature_target: b, embed_size]
            embed_am --> feature_am[feature_am: b, embed_size]
            re_embed_img --> feature_img[feature_img: b, embed_size]
            embed_action --> feature_action[feature_action: b, embed_size]
        end

        features_stack[Stack Features]
        feature_lidar --> features_stack
        feature_target --> features_stack
        feature_am --> features_stack
        feature_img --> features_stack
        feature_action --> features_stack

        subgraph Network
            use_attention[Use Attention?]
            net[Net]
            output_layer[Output Layer]

            features_stack --> use_attention
            use_attention -->|No| simple_net[Simple Net]
            use_attention -->|Yes| attention_net[AttentionNetwork]

            subgraph Simple_Net
                simple_layers[Layers]
                simple_output[Output Layer]

                features_stack --> simple_layers
                simple_layers --> simple_output
                simple_output --> simple_net
            end

            subgraph Attention_Net
                attention_encoder[Transformer Encoder]
                attention_rearrange[Rearrange]
                attention_output[Sequential Output]

                features_stack --> attention_encoder
                attention_encoder --> attention_rearrange
                attention_rearrange --> attention_output
                attention_output --> attention_net
            end

            net --> output_layer
        end

        output[out: b, output_size]
        output_layer --> output
    end
```

This repository contains code for the paper [HOPE: A Reinforcement Learning-based Hybrid Policy Path Planner for Diverse Parking Scenarios](https://arxiv.org/abs/2405.20579). This work proposes a novel solution to the path-planning task in parking scenarios. The planner integrates a reinforcement learning agent with Reeds-Shepp curves, enabling effective planning across diverse scenarios. HOPE guides the exploration of the reinforcement learning agent by applying an action mask mechanism and employs a transformer to integrate the perceived environmental information with the mask. Our approach achieved higher planning success rates compared with typical rule-based algorithms and traditional reinforcement learning methods, especially in challenging cases.

## Examples
### Simulation cases
![simulation](assets/examples.jpg)

### Realworld demo
[https://www.youtube.com/watch?v=62w9qhjIuRI](https://www.youtube.com/watch?v=62w9qhjIuRI)
![realworld](assets/realworld-cases.jpg)

## Setup
1. Install conda or miniconda

2. Clone the repo and build the environment
```Shell
git clone https://github.com/jiamiya/HOPE.git
cd HOPE
conda create -n HOPE python==3.8
conda activate HOPE
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
and install pytorch from [https://pytorch.org/](https://pytorch.org/).

## Usage
### Run a pre-trained agent
```Shell
cd src

python ./evaluation/eval_mix_scene.py ./model/ckpt/HOPE_SAC0.pt --eval_episode 10 --visualize True

python ./evaluation/eval_mix_scene.py ./model/ckpt/HOPE_PPO.pt --eval_episode 50 --visualize True
```
You can find some other pre-trained weights in ``./src/model/ckpt``.

### Train the HOPE planner
```Shell
cd src
python ./train/train_HOPE_sac.py
```
or
```Shell
python ./train/train_HOPE_ppo.py
```

## Citation
If you find our work useful, please cite us as
```bibtex
@article{jiang2024hope,
  title={HOPE: A Reinforcement Learning-based Hybrid Policy Path Planner for Diverse Parking Scenarios},
  author={Jiang, Mingyang and Li, Yueyuan and Zhang, Songan and Chen, Siyuan and Wang, Chunxiang and Yang, Ming},
  journal={arXiv preprint arXiv:2405.20579},
  year={2024}
}
```

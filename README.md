# GraphLayoutLM

## Installation

```
git clone https://github.com/Line-Kite/GraphLayoutLM
cd GraphLayoutLM
conda create -n graphlayoutlm python=3.7
conda activate graphlayoutlm
pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
pip install -r requirements.txt
```


## Pre-trained Models
| Model               | Model Name (Path)                                                                                              | 
|---------------------|----------------------------------------------------------------------------------------------------------------|
| graphlayoutlm-base  | [graphlayoutlm-base](https://drive.google.com/drive/folders/1KV2r4crHcGoTKM7DvEIN6BWEMEdV9tIZ?usp=drive_link)  |
| graphlayoutlm-large | [graphlayoutlm-large](https://drive.google.com/drive/folders/1-zM5L34quKQwfvROvlK7UJWU6HKAGAmF?usp=drive_link) |


## Finetuning Examples

## CORD


  |Model on CORD                                                                                                                | precision | recall |    f1    | accuracy |
  |:---------------------------------------------------------------------------------------------------------------------------:|:---------:|:------:|:--------:|:--------:|
  | [graphlayout-base-finetuned-cord](https://drive.google.com/drive/folders/1F593PVKVGFIfpJyRSiMZmZywZevmlKhs?usp=drive_link)  |   0.9724  | 0.9760 |  0.9742  |  0.9813  |
  | [graphlayout-large-finetuned-cord](https://drive.google.com/drive/folders/1ZZzxG2qDnkoiADovZLovIxdhwwuqJfPc?usp=drive_link) |   0.9791  | 0.9805 |  0.9798  |  0.9839  |

### finetune

Download the model weights and move it to a new directory named "pretrained".

Download the [CORD](https://drive.google.com/drive/folders/14OEWr86qotVBMAsWk7lymMytxn5u-kM6) dataset and move it to a new directory named "datasets".

#### base

```
cd examples
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 20655 run_cord.py \
    --dataset_name cord \
    --do_train \
    --do_eval \
    --model_name_or_path ../pretrained/graphlayoutlm-base  \
    --output_dir ../path/cord/base/test \
    --segment_level_layout 1 --visual_embed 1 --input_size 224 \
    --max_steps 2000 --save_steps -1 --evaluation_strategy steps --eval_steps 100 \
    --learning_rate 5e-5 --per_device_train_batch_size 2 --gradient_accumulation_steps 1 \
    --dataloader_num_workers 8 --overwrite_output_dir
```

#### large

```
cd examples
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 20655 run_cord.py \
    --dataset_name cord \
    --do_train \
    --do_eval \
    --model_name_or_path ../pretrained/graphlayoutlm-large  \
    --output_dir ../path/cord/large/test \
    --segment_level_layout 1 --visual_embed 1 --input_size 224 \
    --max_steps 4000 --save_steps -1 --evaluation_strategy steps --eval_steps 100 \
    --learning_rate 5e-5 --per_device_train_batch_size 2 --gradient_accumulation_steps 1 \
    --dataloader_num_workers 8 --overwrite_output_dir
```


## Citation
Please cite our paper if the work helps you.
```
@inproceedings{li2023enhancing,
  title={Enhancing Visually-Rich Document Understanding via Layout Structure Modeling},
  author={Li, Qiwei and Li, Zuchao and Cai, Xiantao and Du, Bo and Zhao, Hai},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={4513--4523},
  year={2023}
}
```


## Note

We will follow-up complement other examples.

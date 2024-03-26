## About:
Official code for KBS 2024 paper: 
SelectE: Multi-scale adaptive selection network for knowledge graph representation learning

## ZhiHu:
https://zhuanlan.zhihu.com/p/689176699

## Abstract:
Most knowledge graphs in the real world suffer from incompleteness which can be addressed through knowledge graph representation learning (KGRL) techniques that use known facts to infer missing links. In this paper, a novel multi-scale adaptive selection network for KGRL, namely SelectE, is developed to learn richer multi-scale interactive features and automatically select important features, thereby achieving promising KGRL performance. Specifically, first, the input feature matrix is redesigned to better cooperate with multi-scale convolution to improve the interaction of entities and relations. Second, a multi-scale learning module is designed to learn richer multi-scale features from the input matrix using multiple branches with different kernel sizes. Third, to automatically strengthen the important features and suppress the useless features, a multi-scale adaptive selection mechanism is proposed to dynamically allocate the weights of the obtained features based on their contained information. The core of SelectE is to maximize interactions while also considering how to better utilize features. Finally, the outstanding performance of SelectE is validated by a series of comparison experiments on seven benchmark datasets (FB15k-237, WN18RR, FB15k, WN18, YAGO3–10, KINSHIP, UMLS). The experimental results show that SelectE outperforms other state-of-the-art models, demonstrating its remarkable performance and generalization.

## Paper:
SelectE: Multi-scale adaptive selection network for knowledge graph representation learning.

https://www.sciencedirect.com/science/article/abs/pii/S0950705124001898

## Requirements:
To reproduce the results, 
1) pip install -r requirements.txt
2) run final.sh:
  ```shell
#! /bin/bash
python_env=python # python in env

# WN18RR 
$python_env SelectE.py --data_path "./data" --run_folder "./" --data_name "WN18RR" --embedding_dim 200 --filter1_size 1 3 --filter2_size 3 3 --filter3_size 1 5 --output_channel 20 --min_lr 0.00001 --batch_size 1500 --log_epoch 2 --neg_ratio 1 --input_drop 0.2 --hidden_drop 0.5 --feature_map_drop 0.2 --opt "Adam" --learning_rate 0.001 --weight_decay 5e-4 --factor 0.5 --verbose 1 --patience 5 --max_mrr 0 --epoch 300 --momentum 0.9 --save_name "./model/wn18rr.pt"
## FB15K-237
#$python_env SelectE.py --data_path "./data" --run_folder "./" --data_name "FB15k-237" --embedding_dim 200 --filter1_size 1 3 --filter2_size 3 3 --filter3_size 1 5 --output_channel 32 --min_lr 0.000005 --batch_size 2000 --log_epoch 2 --neg_ratio 1 --input_drop 0.2 --hidden_drop 0.4 --feature_map_drop 0.3 --opt "Adam" --learning_rate 0.0005 --weight_decay 5e-4 --factor 0.5 --verbose 1 --patience 5 --max_mrr 0 --epoch 800 --momentum 0.9 --save_name "./model/FB15K237.pt"
## KINSHIP
#$python_env SelectE.py --data_path "./data" --run_folder "./" --data_name "KINSHIP" --embedding_dim 200 --filter1_size 1 3 --filter2_size 3 3 --filter3_size 1 5 --output_channel 20 --min_lr 0.00001 --batch_size 800 --log_epoch 2 --neg_ratio 1 --input_drop 0.3 --hidden_drop 0.1 --feature_map_drop 0.4 --opt "Adam" --learning_rate 0.001 --weight_decay 5e-3 --factor 0.5 --verbose 1 --patience 5 --max_mrr 0 --epoch 220 --momentum 0.9 --save_name "./model/KINSHIP.pt"
## UMLS
#$python_env SelectE.py --data_path "./data" --run_folder "./" --data_name "UMLS" --embedding_dim 200 --filter1_size 1 3 --filter2_size 3 3 --filter3_size 1 5 --output_channel 24 --min_lr 0.00001 --batch_size 900 --log_epoch 2 --neg_ratio 1 --input_drop 0.2 --hidden_drop 0.1 --feature_map_drop 0.3 --opt "Adam" --learning_rate 0.001 --weight_decay 5e-3 --factor 0.5 --verbose 1 --patience 5 --max_mrr 0 --epoch 400 --momentum 0.9 --save_name "./model/UMLS.pt"
## WN18
#$python_env SelectE.py --data_path "./data" --run_folder "./" --data_name "WN18" --embedding_dim 200 --filter1_size 1 3 --filter2_size 3 3 --filter3_size 1 5 --output_channel 20 --min_lr 0.00001 --batch_size 1500 --log_epoch 2 --neg_ratio 1 --input_drop 0.3 --hidden_drop 0.5 --feature_map_drop 0.1 --opt "Adam" --learning_rate 0.0003 --weight_decay 5e-8 --factor 0.5 --verbose 1 --patience 5 --max_mrr 0 --epoch 1000 --momentum 0.9 --save_name "./model/WN18.pt"
## yago
#$python_env SelectE.py --data_path "./data" --run_folder "./" --data_name "YAGO3-10" --embedding_dim 200 --filter1_size 1 3 --filter2_size 3 3 --filter3_size 1 5 --output_channel 20 --min_lr 0.00001 --batch_size 1500 --log_epoch 2 --neg_ratio 1 --input_drop 0.2 --hidden_drop 0.1 --feature_map_drop 0.3 --opt "Adam" --learning_rate 0.001 --weight_decay 5e-9 --factor 0.5 --verbose 1 --patience 5 --max_mrr 0 --epoch 350 --momentum 0.9 --save_name "./model/yago.pt"
## FB15K
#$python_env SelectE.py --data_path "./data" --run_folder "./" --data_name "FB15k" --embedding_dim 200 --filter1_size 1 3 --filter2_size 3 3 --filter3_size 1 5 --output_channel 20 --min_lr 0.00001 --batch_size 1500 --log_epoch 2 --neg_ratio 1 --input_drop 0.2 --hidden_drop 0.1 --feature_map_drop 0.3 --opt "Adam" --learning_rate 0.001 --weight_decay 5e-8 --factor 0.5 --verbose 1 --patience 5 --max_mrr 0 --epoch 500 --momentum 0.9 --save_name "./model/FB15K.pt"
  ```

## Citation:
@article{zu2024selecte,
  title={SelectE: Multi-scale adaptive selection network for knowledge graph representation learning},
  author={Zu, Lizheng and Lin, Lin and Fu, Song and Guo, Feng and Wu, Jinlei},
  journal={Knowledge-Based Systems},
  pages={111554},
  year={2024},
  publisher={Elsevier}
}

## Acknowledgement
This repo benefits from ConvE,DeepE,M_DCN,InteractE and ConvR. Thanks for their wonderful works.

ConvE: https://github.com/TimDettmers/ConvE

DeepE: https://github.com/zhudanhao/DeepE

M_DCN: https://github.com/zhifei1993/M_DCN

InteractE: https://github.com/malllabiisc/InteractE

ConvR: https://github.com/shuibinlong/ConvR

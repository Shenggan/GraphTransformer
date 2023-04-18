# GraphTransformer

CS 6208 Group Project

### Prepare Dataset

```bash
wget https://github.com/daiquocnguyen/Graph-Transformer/raw/master/dataset.zip
unzip dataset.zip
```

#### Requirements
- Python 	3.x
- PyTorch	1.13
- networkx 	3.0
- scikit-learn	1.2.2

### Usage

For UGformerV1

```bash
cd UGformerV1
python train_UGformerV1_Sup.py --dataset IMDBBINARY --batch_size 4 --ff_hidden_size 1024 --fold_idx 1 --num_neighbors 8 --num_epochs 50 --num_timesteps 4 --learning_rate 0.0005 --model_name IMDBBINARY_bs4_fold1_1024_8_idx0_4_1
```

For UGformerV2 with GGAT, please see `./UGformerV2/README.md`

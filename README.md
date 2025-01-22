# MultiTrans
This repository includes the official project of our paper submitted to Computer Methods and Programs in Biomedicine. Title: "MultiTrans: Multi-Branch Transformer Network for Medical Image Segmentation".

## Usage

### 0. To be noted:

- We will gradually optimize the code to make it more readable and standardized.

- If you have any suggestions for improvement or encounter any issues while using this code, please feel free to contact me: yanhuazhang@mail.nwpu.edu.cn

### 1. Download pre-trained Resnet models

Download the pre-trained Resnet models and put them into the folder 'pre_trained_Resnet'.

- resnet50-deep-stem:[link](https://drive.google.com/file/d/1OktRGqZ15dIyB2YTySLfOVtprerHgbef/view?usp=sharing)

- resnet50:[link](https://drive.google.com/file/d/1fUAuRfewRpaS5mFX_IQqrE2syEn9PXrv/view?usp=sharing)

- resnet34:[link](https://drive.google.com/file/d/18Erx_ISMt1XMjJlgl4SQsr-iMvcN-7bZ/view?usp=sharing)

- resnet18-deep-stem:[link](https://drive.google.com/file/d/1q1VBV37acIte0GynoS054BWfwwdx1NiZ/view?usp=sharing)

- resnet18:[link](https://drive.google.com/file/d/1LCybGjJ_d-nALvciBBkZil_XfO-7ptAE/view?usp=sharing)

### 2. Prepare data

Download the preprocessed data and put it into the folder 'preprocessed_data'.

- Download the Synapse dataset from [official website](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789). Convert them to numpy format, clip within [-125, 275], normalize each 3D volume to [0, 1], and extract 2D slices from 3D volume for training while keeping the testing 3D volume in h5 format.

- Or directly use [preprocessed data](https://drive.google.com/file/d/1XjHzJageFKFN7Tg-6F2NJz2sj9hSLPK0/view?usp=sharing) provided by [TransUNet](https://github.com/Beckschen/TransUNet).

### 3. Environment

We trained our model in two different environments:

one NVIDIA GeForce GTX 3090 (24GB) with the CUDA 11.1 and CUDNN 8.0.

- Python 3.8.13.

- PyTorch 1.8.1. 

one NVIDIA A800 (80GB) with the CUDA 11.7 and CUDNN 8500.

- Python 3.8.13.

- PyTorch 2.0.1. 

Please refer to 'requirements.txt' for other dependencies.

### 4. Train/Test by yourself

```bash
cd MultiTrans_extension
```

- Run the train script.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --Model_Name My_Model --bran_weights 0.4 0.3 0.2 0.1 --base_lr 0.1 --branch_depths 5 5 5 5 5 --branch_in_channels 256 256 256 256 256 --branch_key_channels 32 32 32 32 32 --Self_Attention_Name='ESA_MultiTrans' --seed 1294
```

- Run the test script.

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --dataset Synapse --Model_Name My_Model --bran_weights 0.4 0.3 0.2 0.1 --base_lr 0.1 --branch_depths 5 5 5 5 5 --branch_in_channels 256 256 256 256 256 --branch_key_channels 32 32 32 32 32 --Self_Attention_Name='ESA_MultiTrans' --seed 1294 --is_savenii=True
```

### 5. Ablation experiments on the self-attention module.

- Remove the Head-sharing operation.

```bash
cd MultiTrans_extension
```

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --Model_Name My_Model --bran_weights 0.4 0.3 0.2 0.1 --base_lr 0.1 --branch_depths 5 5 5 5 5 --branch_in_channels 256 256 256 256 256 --branch_key_channels 32 32 32 32 32 --Self_Attention_Name='ESA_MultiTrans' --seed 1294 --one_kv_head='False' --marker='No_HeadShare'
```

- Remove the Projection-sharing operation.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --Model_Name My_Model --bran_weights 0.4 0.3 0.2 0.1 --base_lr 0.1 --branch_depths 5 5 5 5 5 --branch_in_channels 256 256 256 256 256 --branch_key_channels 32 32 32 32 32 --Self_Attention_Name='ESA_MultiTrans' --seed 1294 --share_kv='False' --marker='No_ProjectionShare'
```

- Use the Standard self-attention.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --Model_Name My_Model --bran_weights 0.4 0.3 0.2 0.1 --base_lr 0.1 --branch_depths 5 5 5 5 5 --branch_in_channels 256 256 256 256 256 --branch_key_channels 32 32 32 32 32 --Self_Attention_Name='SSA' --one_kv_head False --share_kv False --seed 1294
```

## Reference
* [TransUNet](https://github.com/Beckschen/TransUNet)

## Citations

```bibtex

xxx

```

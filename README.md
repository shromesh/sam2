# sam2 experiments

## setup

### environment
- デモ: https://sam2.metademolab.com/
- ダウンロード: https://github.com/shromesh/sam2
- 論文: https://arxiv.org/abs/2408.00714

```
pyenv install 3.11.11 && \
cd sam2 && \
pyenv local 3.11.11 && \
python -m venv .venv && \
source .venv/bin/activate && \
pip install -e . && \
pip install -e ".[notebooks]" && \
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

### notebookを動かす
`notebooks/image_predictor_example.ipynb`

## fine-tuning
sam2/my_sam2_experiment/
  ├─ images/
  │   └─ png, jpg
  ├─ masks/
  │   └─ png, jpg
  ├─ train.csv
  └─ finetune_sam2.py
  ├─ sam2.1_hiera_t.yaml
  └─ sam2.1_hiera_tiny.pt
を用意

- sam2/sam2/内に `sam2.1_hiera_t.yaml` を追加

Python 3.10.16

```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install opencv-python matplotlib scikit-learn pandas
```

```
cd my_sam2_experiment
python finetune_sam2.py
```
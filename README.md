# sam2 experiments

## local setup

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

### 
my_sam2_experiment/内に
- sam2.1_hiera_t.yaml
- sam2.1_hiera_tiny.pt
- train.csv
- images/
- masks/
- 学習ファイル（python）
を用意

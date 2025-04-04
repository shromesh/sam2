# sam2 experiments

## local setup

### environment
- デモ: https://sam2.metademolab.com/
- ダウンロード: https://github.com/shromesh/sam2
- 論文: https://arxiv.org/abs/2408.00714

```
git clone git@github.com:facebookresearch/sam2.git
cd sam2
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e ".[notebooks]"
```
`pip install -e .`によってsetup.pyが実行される。

### SAM 2モデルのダウンロード
```
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

### notebookを動かす
`notebooks/image_predictor_example.ipynb`

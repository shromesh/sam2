# sam2 experiments

## local setup

### environment
- デモ: https://sam2.metademolab.com/
- ダウンロード: https://github.com/shromesh/sam2
- 論文: https://arxiv.org/abs/2408.00714

```
cd sam2
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e ".[notebooks]"
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

### notebookを動かす
`notebooks/image_predictor_example.ipynb`

# finetune_sam2.py

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# segment-anything-2 のモジュールをインポート
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ====== 設定パラメータ ======
DATA_DIR = "."
IMAGES_DIR = os.path.join(DATA_DIR, "images")
MASKS_DIR = os.path.join(DATA_DIR, "masks")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")

# 事前学習済み SAM2 のパス
CHECKPOINT_PATH = "sam2.1_hiera_tiny.pt"
MODEL_CFG = "sam2.1_hiera_t.yaml"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_STEPS = 10  # デモとして学習を10ステップだけ回す


def main():
    # ==============================
    # 1) データ読み込み
    # ==============================
    import pandas as pd

    df = pd.read_csv(TRAIN_CSV)
    train_data = []
    for _, row in df.iterrows():
        train_data.append(
            {
                "image": os.path.join(IMAGES_DIR, row["ImageId"]),
                "annotation": os.path.join(MASKS_DIR, row["MaskId"]),
            }
        )

    # ==============================
    # 2) SAM2モデルのロード
    # ==============================
    model = build_sam2(MODEL_CFG, CHECKPOINT_PATH, device=DEVICE)
    predictor = SAM2ImagePredictor(model)

    # 学習対象: mask_decoder と prompt_encoder のみ
    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        params=predictor.model.parameters(), lr=1e-4, weight_decay=1e-4
    )

    # ==============================
    # 3) 学習ループ (超簡易版)
    # ==============================
    for step in range(1, NUM_STEPS + 1):
        # 1枚しかないので常に同じ画像を読む (random でも何でもOK)
        data_item = train_data[0]
        img_path = data_item["image"]
        msk_path = data_item["annotation"]

        # 画像読み込み (BGR->RGB)
        img_bgr = cv2.imread(img_path)
        # img_rgb = img_bgr[..., ::-1]
        # import numpy as np
        img_rgb = np.ascontiguousarray(img_bgr[..., ::-1])

        # マスク(グレイスケール)
        gt_mask_gray = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)

        # リサイズ (大きい場合は適宜)
        h, w = img_rgb.shape[:2]
        if max(h, w) > 1024:
            r = 1024 / max(h, w)
            new_w = int(w * r)
            new_h = int(h * r)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h))
            gt_mask_gray = cv2.resize(
                gt_mask_gray, (new_w, new_h), interpolation=cv2.INTER_NEAREST
            )

        # 二値化 (画素値 > 0 を前景とみなす)
        binary_mask = (gt_mask_gray > 0).astype(np.uint8)

        # 適当に1点をマスク内部からサンプリング (プロンプトとして)
        coords = np.argwhere(binary_mask == 1)
        if len(coords) == 0:
            print("マスクが空でした。")
            continue
        # ランダムに1点だけ取得
        pt = coords[np.random.randint(len(coords))]
        input_point = np.array([[[pt[1], pt[0]]]])  # (B,1,2) = (x,y)
        input_label = np.ones((1, 1), dtype=np.uint8)  # ラベル=1

        # --------------------------------------
        # SAM2 に画像をセット
        predictor.set_image(img_rgb)

        # Prompt Encoder に座標を与えて埋め込み取得
        # (内部関数 _prep_prompts はpublic APIではないので注意)
        mask_input, unnorm_coords, labels, _ = predictor._prep_prompts(
            input_point,
            input_label,
            box=None,
            mask_logits=None,
            normalize_coords=True,
        )

        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=(unnorm_coords, labels), boxes=None, masks=None
        )

        # Mask Decoder 出力
        high_res_features = [
            feat_level[-1].unsqueeze(0)
            for feat_level in predictor._features["high_res_feats"]
        ]
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        # 後処理(リサイズ)
        prd_masks = predictor._transforms.postprocess_masks(
            low_res_masks, predictor._orig_hw[-1]
        )  # shape: (B=1, num_masks=3, H, W)のイメージ

        # ------------------------------------------------
        # 損失計算 (単純な二値クロスエントロピー + スコア誤差)
        gt_mask_tensor = torch.tensor(
            binary_mask[None, None, ...], dtype=torch.float32
        ).to(DEVICE)
        pred_mask = torch.sigmoid(prd_masks[:, 0])  # 3系統のうち1番目だけ使う
        seg_loss = -(
            gt_mask_tensor * torch.log(pred_mask + 1e-6)
            + (1 - gt_mask_tensor) * torch.log(1 - pred_mask + 1e-6)
        ).mean()

        # IOU 計算してスコアロス (あまり気にしなくてもOK)
        inter = (gt_mask_tensor * (pred_mask > 0.5)).sum()
        union = (gt_mask_tensor + (pred_mask > 0.5)).clamp(0, 1).sum()
        iou = inter / (union + 1e-6)
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

        loss = seg_loss + 0.05 * score_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1 == 0:
            print(f"[Step {step}] loss={loss.item():.4f}, IoU={iou.item():.3f}")

    # 学習後のモデルを保存
    torch.save(predictor.model.state_dict(), "fine_tuned_sam2_one_sample.pt")
    print("Done. 重みを fine_tuned_sam2_one_sample.pt に保存しました。")


if __name__ == "__main__":
    main()

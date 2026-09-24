import torch
from pathlib import Path
from visual_bge.visual_bge.modeling import Visualized_BGE

project_root = Path(__file__).resolve().parents[2]
model_path = project_root / "models" / "bge" / "Visualized_base_en_v1.5.pth"
image_dir = project_root / "data" / "C3" / "imgs"

if not model_path.is_file():
    raise FileNotFoundError(
        f"缺少模型权重：{model_path}\n"
        "请先在 code/C3 目录运行 python download_model.py 下载模型。"
    )

model = Visualized_BGE(model_name_bge="BAAI/bge-base-en-v1.5",
                      model_weight=str(model_path))
model.eval()

with torch.no_grad():
    text_emb = model.encode(text="datawhale开源组织的logo")
    img_emb_1 = model.encode(image=str(image_dir / "datawhale01.png"))
    multi_emb_1 = model.encode(image=str(image_dir / "datawhale01.png"), text="datawhale开源组织的logo")
    img_emb_2 = model.encode(image=str(image_dir / "datawhale02.png"))
    multi_emb_2 = model.encode(image=str(image_dir / "datawhale02.png"), text="datawhale开源组织的logo")

# 计算相似度
sim_1 = img_emb_1 @ img_emb_2.T
sim_2 = img_emb_1 @ multi_emb_1.T
sim_3 = text_emb @ multi_emb_1.T
sim_4 = multi_emb_1 @ multi_emb_2.T

print("=== 相似度计算结果 ===")
print(f"纯图像 vs 纯图像: {sim_1}")
print(f"图文结合1 vs 纯图像: {sim_2}")
print(f"图文结合1 vs 纯文本: {sim_3}")
print(f"图文结合1 vs 图文结合2: {sim_4}")

# 向量信息分析
print("\n=== 嵌入向量信息 ===")
print(f"多模态向量维度: {multi_emb_1.shape}")
print(f"图像向量维度: {img_emb_1.shape}")
print(f"多模态向量示例 (前10个元素): {multi_emb_1[0][:10]}")
print(f"图像向量示例 (前10个元素):   {img_emb_1[0][:10]}")

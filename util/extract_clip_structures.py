import os
from pathlib import Path
import open_clip
import torch

# 模型列表
model_list = [
    ["RN50", "openai"],
    ["RN101", "openai"],
    ["RN50x4", "openai"],
    ["RN50x64", "openai"],
    ["ViT-B-32", "openai"],
    ["ViT-B-32-quickgelu", "laion400m_e31"],
    ["ViT-B-16", "openai"],
    ["ViT-B-16-plus-240", "laion400m_e31"],
    ["ViT-L-14", "openai"],
    ["ViT-L-14-quickgelu", "metaclip_fullcc"],
    ["ViT-L-14-336", "openai"],
    ["ViT-H-14", "laion2b_s32b_b79k"],
    ["ViT-g-14", "laion2b_s12b_b42k"],
    ["ViT-bigG-14", "laion2b_s39b_b160k"],
    ["roberta-ViT-B-32", "laion2b_s12b_b32k"],
    ["xlm-roberta-base-ViT-B-32", "laion5b_s13b_b90k"],
    ["convnext_base_w", "laion2b_s13b_b82k"],
    ["convnext_large_d_320", "laion2b_s29b_b131k_ft"],
    ["convnext_xxlarge", "laion2b_s34b_b82k_augreg_soup"],
    ["coca_ViT-B-32", "laion2b_s13b_b90k"],
    ["EVA01-g-14", "laion400m_s11b_b41k"],
    ["EVA02-B-16", "merged2b_s8b_b131k"],
    ["nllb-clip-base-siglip", "v1"]
]



# 输出目录
output_dir = Path("clip_model_structures")
output_dir.mkdir(exist_ok=True)

# 遍历模型列表
for model_name, pretrained in model_list:
    print(f"\n🔹 正在处理模型: {model_name} ({pretrained}) ...")
    try:
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model.eval()
        structure_text = str(model)
        file_path = output_dir / f"{model_name}_{pretrained}.txt"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Model: {model_name} ({pretrained})\n")
            f.write("=" * 80 + "\n\n")
            f.write(structure_text)

        print(f"✅ 模型结构已保存: {file_path.name}")
    except Exception as e:
        error_path = output_dir / f"{model_name}_{pretrained}_ERROR.txt"
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(f"⚠️ 加载模型失败: {model_name} ({pretrained})\n\n错误信息:\n{e}\n")
        print(f"❌ 加载失败，错误已记录: {error_path.name}")

print("\n全部模型处理完成 ✅")
print(f"输出路径: {output_dir.resolve()}")

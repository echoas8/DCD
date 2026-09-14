import logging
import torch.nn as nn

def extract_visual_layers(model):
    """
    通用视觉层提取器（兼容 open_clip 的视觉编码器）

    支持：
      - ViT (visual.transformer.resblocks)
      - ModifiedResNet (visual.layer1~4 + attnpool)
      - ConvNeXt (visual.trunk.stages[*].blocks[*])
      - EVA (visual.trunk.patch_embed + blocks + norm + head)
      - VisionTransformer + AttentionPoolLatent
        (visual.trunk.patch_embed + blocks + norm + attn_pool + fc_norm + head)
      - 其它 Hybrid ViT (visual.trunk.blocks)

    返回:
        layers: List[nn.Module]
        layer_names: List[str]
    """
    visual = model.visual
    layers, layer_names = [], []

    # =====  ViT 系列（CLIP 原生 ViT）=====
    if hasattr(visual, "transformer"):
        for i, blk in enumerate(visual.transformer.resblocks):
            layers.append(blk)
            layer_names.append(f"resblock_{i}")
        logging.info(f"Detected VisionTransformer with {len(layers)} blocks.")

    # =====  ResNet 系列 =====
    elif hasattr(visual, "layer1") and hasattr(visual, "attnpool"):
        for name in ["layer1", "layer2", "layer3", "layer4"]:
            if hasattr(visual, name):
                layers.append(getattr(visual, name))
                layer_names.append(name)
        layers.append(visual.attnpool)
        layer_names.append("attnpool")
        logging.info(f"Detected ModifiedResNet with {len(layers)} layers.")

    # =====  ConvNeXt 系列（timm）=====
    elif hasattr(visual, "trunk") and hasattr(visual.trunk, "stages"):
        trunk = visual.trunk

        # 逐个 stage / block 展开
        for s_idx, stage in enumerate(trunk.stages):
            # 下采样（stage0 通常 Identity，可跳过）
            if hasattr(stage, "downsample"):
                ds = stage.downsample
                if not isinstance(ds, nn.Identity):
                    layers.append(ds)
                    layer_names.append(f"stage{s_idx}_downsample")

            # ConvNeXtBlock 序列
            if hasattr(stage, "blocks"):
                for b_idx, blk in enumerate(stage.blocks):
                    layers.append(blk)
                    layer_names.append(f"stage{s_idx}_block{b_idx}")

        logging.info(f"Detected ConvNeXt with {len(layers)} layers.")

    # =====  EVA 系列=====
    elif hasattr(visual, "trunk") and hasattr(visual.trunk, "blocks") and \
         "eva" in type(visual.trunk).__name__.lower():
        trunk = visual.trunk

        # EvaBlock 序列
        for i, blk in enumerate(trunk.blocks):
            layers.append(blk)
            layer_names.append(f"block_{i}")

        # head
        if hasattr(trunk, "head"):
            layers.append(trunk.head)
            layer_names.append("head")

        logging.info(
            f"Detected EVA backbone ({type(trunk).__name__}) "
            f"with {len(layers)} layers."
        )

    # =====  VisionTransformer + AttentionPoolLatent（如 nllb-clip）=====
    elif hasattr(visual, "trunk") and hasattr(visual.trunk, "blocks") and (
         "visiontransformer" in type(visual.trunk).__name__.lower()
         or hasattr(visual.trunk, "attn_pool")):
        trunk = visual.trunk

        # Transformer Block 序列
        for i, blk in enumerate(trunk.blocks):
            layers.append(blk)
            layer_names.append(f"block_{i}")

        # AttentionPoolLatent
        if hasattr(trunk, "attn_pool"):
            layers.append(trunk.attn_pool)
            layer_names.append("attn_pool")

        # 外层 visual.head（一般是 Sequential() 空头，可以按需过滤）
        if hasattr(visual, "head") and not isinstance(visual.head, nn.Identity):
            layers.append(visual.head)
            layer_names.append("clip_visual_head")

        logging.info(
            f"Detected VisionTransformer backbone ({type(trunk).__name__}) "
            f"with {len(layers)} layers (with attn_pool)."
        )

    # ===== 6. 通用 Hybrid ViT 结构 =====
    elif hasattr(visual, "trunk") and hasattr(visual.trunk, "blocks"):
        trunk = visual.trunk
        for i, blk in enumerate(trunk.blocks):
            layers.append(blk)
            layer_names.append(f"block_{i}")
        logging.info(
            f"Detected Hybrid Vision Transformer ({type(trunk).__name__}) "
            f"with {len(layers)} blocks."
        )

    else:
        logging.warning(f"Unknown visual encoder type: {type(visual)}")

    return layers, layer_names

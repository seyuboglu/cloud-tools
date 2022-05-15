from ..config import *


def oxford_test_convnext_unet():

    sweep = prod(
        [
            flag("dataset", ["oxfordpets"]),
            flag("model", ["convnext_unet_tiny"]),
            flag("encoder", ["null"]),
            flag("decoder", ["id"]),
            flag("task.loss", ["focal_loss", "binary_cross_entropy"]),
            flag("task.metrics", [["binary_accuracy", "accuracy", "iou_with_logits"]]),
        ]
    )

    return sweep
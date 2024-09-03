from typing import Any, Dict

from torch import nn

__all__ = ["SparseRefiner"]


class SparseRefiner(nn.Module):
    def __init__(
        self,
        selector: nn.Module,
        featurizer: nn.Module,
        backbone: nn.Module,
        classifier: nn.Module,
        ensembler: nn.Module,
    ) -> None:
        super().__init__()
        self.selector = selector
        self.featurizer = featurizer
        self.backbone = backbone
        self.classifier = classifier
        self.ensembler = ensembler

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        mask = self.selector(inputs)
        x = self.featurizer(inputs, mask)
        x = self.backbone(x).F
        yo = self.classifier(x)

        logits = inputs["logits"]
        yi = logits[mask]
        ye = self.ensembler(yi, yo)

        outputs = {}
        outputs["logits/i/mask"] = yi
        outputs["logits/o/mask"] = yo
        outputs["logits/e/mask"] = ye

        for name in ["logits/i", "logits/o", "logits/e"]:
            y = logits.clone()
            y[mask] = outputs[name + "/mask"].to(dtype=y.dtype)
            outputs[name + "/full"] = y

        if "label" in inputs:
            outputs["label/full"] = inputs["label"]
            outputs["label/mask"] = inputs["label"][mask]

        return outputs
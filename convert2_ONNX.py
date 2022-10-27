from collections import OrderedDict
import torch
from torch.onnx import TrainingMode

def convert_to_onnx(
    model_pytorch, output_path: str, inputs_pytorch, opset: int = 12
) -> None:
    """
    Convert a Pytorch model to an ONNX graph by tracing the provided input inside the Pytorch code.
    :param model_pytorch: Pytorch model
    :param output_path: where to save ONNX file
    :param inputs_pytorch: Tensor, can be dummy data, shape is not important as we declare all axes as dynamic.
    Should be on the same device than the model (CPU or GPU)
    :param opset: version of ONNX protocol to use, usually 12, or 13 if you use per channel quantized model
    """
    # dynamic axis == variable length axis
    dynamic_axis = OrderedDict()
    for k in inputs_pytorch.keys():
        dynamic_axis[k] = {0: "batch_size", 1: "sequence"}
    dynamic_axis["output"] = {0: "batch_size"}
    with torch.no_grad():
        torch.onnx.export(
            model_pytorch,  # model to optimize
            args=tuple(inputs_pytorch.values()),  # tuple of multiple inputs
            f=output_path,  # output path / file object
            opset_version=opset,  # the ONNX version to use, 13 if quantized model, 12 for not quantized ones
            do_constant_folding=True,  # simplify model (replace constant expressions)
            input_names=list(inputs_pytorch.keys()),  # input names
            output_names=["output"],  # output axis name
            dynamic_axes=dynamic_axis,  # declare dynamix axis for each input / output
            training=TrainingMode.EVAL,  # always put the model in evaluation mode
            verbose=False,
        )

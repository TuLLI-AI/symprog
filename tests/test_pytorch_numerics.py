# type: ignore 
import pytest

torch = pytest.importorskip("torch")
import numpy as np
import torch.nn.functional as F

from symprog.solvers.numerical_interfaces.pytorch import PytorchWeights, PytorchWeightsFast

def test_pytorch_config_and_assignment_cpu():
    w = PytorchWeights(weights=[0.1, 0.9], wmap={0:0, 1:1}, device="cpu", requires_grad=True)
    params = w.parameters["weights"]
    assert isinstance(params, torch.Tensor)
    assert params.device.type == "cpu"
    assert params.requires_grad is True

    # assignment returns scalar value from tensor params
    assert w.assignment(0).item() == pytest.approx(0.1)
    assert w.assignment(1).item() == pytest.approx(0.9)

def test_pytorch_basic_ops():
    w = PytorchWeights(weights=[0.3, 0.7], wmap={0:0, 1:1}, device="cpu")
    a = w.add(0.2, 0.5)
    m = w.mult(0.2, 0.5)
    d = w.div(0.5, 0.2)
    la = w.log_add(-1.0, -2.0)
    lm = w.log_mult(-1.0, -2.0)
    ld = w.log_div(-1.0, -2.0)

    assert torch.allclose(a, torch.tensor(0.7))
    assert torch.allclose(m, torch.tensor(0.1))
    assert torch.allclose(d, torch.tensor(2.5))
    assert torch.allclose(la, torch.logaddexp(torch.tensor(-1.0), torch.tensor(-2.0)))
    assert torch.allclose(lm, torch.tensor(-3.0))
    assert torch.allclose(ld, torch.tensor(1.0))

    # log / exp / sigmoid / log_sigmoid
    assert torch.allclose(w.log(2.0), torch.log(torch.tensor(2.0)))
    assert torch.allclose(w.exp(2.0), torch.exp(torch.tensor(2.0)))
    assert torch.allclose(w.sigmoid(0.0), torch.sigmoid(torch.tensor(0.0)))
    assert torch.allclose(w.log_sigmoid(0.0), F.logsigmoid(torch.tensor(0.0)))

    # neg / log_neg
    assert torch.allclose(w.neg(0.25), torch.tensor(0.75))
    val_log = torch.tensor(-2.0)
    assert torch.allclose(w.log_neg(val_log), torch.log1p(-torch.exp(val_log)))

    # max/min
    assert torch.allclose(w.max(0.1, 0.9), torch.tensor(0.9))
    assert torch.allclose(w.min(0.1, 0.9), torch.tensor(0.1))

    # constants
    assert torch.isinf(w.inf())
    assert torch.isinf(w.neg_inf()) and (w.neg_inf() < 0)

def test_pytorch_clone_and_device_transfer():
    w = PytorchWeights(weights=[0.4, 0.6], wmap={0:0, 1:1}, device="cpu")
    t = torch.tensor(0.5, device=w.parameters["weights"].device)
    c = w.clone(t)
    assert c is not t and torch.allclose(c, t)

def test_pytorch_fast_ops_require_tensors():
    wf = PytorchWeightsFast(weights=torch.tensor([0.1, 0.9]), wmap={0:0, 1:1}, device="cpu")
    x = torch.tensor(0.2)
    y = torch.tensor(0.3)
    assert torch.allclose(wf.add(x, y), torch.tensor(0.5))
    assert torch.allclose(wf.mult(x, y), torch.tensor(0.06))
    assert torch.allclose(wf.log_add(torch.log(x), torch.log(y)), torch.logaddexp(torch.log(x), torch.log(y)))
    assert torch.allclose(wf.max(x, y), torch.max(x, y))
    assert torch.allclose(wf.min(x, y), torch.min(x, y))
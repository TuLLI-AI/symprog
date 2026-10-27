# type: ignore 
import math
import numpy as np
import pytest

from symprog.solvers.numerical_interfaces.python import PythonWeights

def test_python_basic_ops():
    w = PythonWeights(weights=[0.2, 0.8], wmap={0:0, 1:1})
    assert w.device is None

    # identity
    assert w.identity(0.5) == 0.5

    # add / mult / div
    assert w.add(0.3, 0.4) == pytest.approx(0.7)
    assert w.mult(0.3, 0.4) == pytest.approx(0.12)
    assert w.div(0.6, 0.3) == pytest.approx(2.0)

    # log add (logaddexp), log mult, log div
    assert w.log_add(-1.0, -2.0) == pytest.approx(np.logaddexp(-1.0, -2.0))
    assert w.log_mult(-1.0, -2.0) == pytest.approx(-3.0)
    assert w.log_div(-1.0, -2.0) == pytest.approx(1.0)

    # log / exp / sigmoid / log_sigmoid
    assert w.log(2.0) == pytest.approx(np.log(2.0))
    assert w.exp(2.0) == pytest.approx(np.exp(2.0))
    assert w.sigmoid(0.0) == pytest.approx(0.5)
    assert w.log_sigmoid(0.0) == pytest.approx(np.log(0.5))

    # neg / log_neg (use log domain value to avoid invalid exp)
    assert w.neg(0.25) == pytest.approx(0.75)
    val_log = -2.0  # exp(-2) < 1
    assert w.log_neg(val_log) == pytest.approx(np.log1p(-np.exp(val_log)))

    # max/min
    assert w.max(0.1, 0.9) == 0.9
    assert w.min(0.1, 0.9) == 0.1

    # constants
    assert w.zero() == 0.0
    assert w.one() == 1.0
    assert math.isinf(w.inf())
    assert math.isinf(w.neg_inf()) and w.neg_inf() < 0.0

def test_python_assignment_and_clone():
    w = PythonWeights(weights=[0.2, 0.8], wmap={0: 0, 1: 1})
    assert w.assignment(0) == pytest.approx(0.2)
    assert w.assignment(1) == pytest.approx(0.8)
    n = w.clone(0.3)
    assert n == 0.3

def test_python_tensor_input_is_detached():
    torch = pytest.importorskip("torch")
    t = torch.tensor([0.1, 0.9], requires_grad=True)
    w = PythonWeights(weights=t, wmap={0:0, 1:1})
    # _config_weights detaches to list
    assert isinstance(w.parameters["weights"], list)
    assert w.parameters["weights"] == pytest.approx([0.1, 0.9])
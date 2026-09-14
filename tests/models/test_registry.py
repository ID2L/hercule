"""Tests for concrete-only model discovery (spec 006, sub-spec S01).

Before this fix, `get_available_models()` fell back to `attr.__name__.lower()` for any
`RLModel` subclass that did not declare its own `model_name`, which registered the
abstract `TDModel` as `"tdmodel"` and let `create_model("tdmodel")` raise a confusing
`TypeError` from instantiating an abstract class instead of a clear `ValueError`.
"""

import inspect
from typing import ClassVar

import pytest

import hercule.models.dummy as dummy_module
from hercule.models import RLModel, create_model, get_available_models


@pytest.mark.unit
def test_registry_returns_exactly_the_four_concrete_models() -> None:
    """`tdmodel` (the abstract `TDModel`) must not appear."""
    available = get_available_models()

    assert set(available.keys()) == {"deep_q_learning", "dummy", "simple_q_learning", "simple_sarsa"}
    assert "tdmodel" not in available


@pytest.mark.unit
def test_no_registered_class_is_abstract() -> None:
    for name, cls in get_available_models().items():
        assert not inspect.isabstract(cls), f"{name} -> {cls} is abstract and must not be registered"


@pytest.mark.unit
def test_locally_defined_abstract_subclass_is_not_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    """An abstract `RLModel` subclass imported into a scanned package stays unregistered.

    `PhantomAbstractModel` deliberately implements none of `RLModel`'s abstract methods
    (`act`, `run_epoch`, `predict`, `_export`, `_import`), so Python's ABC machinery
    keeps it abstract even though it declares `model_name` and `supported_spaces`.
    """

    class PhantomAbstractModel(RLModel):
        model_name: ClassVar[str] = "phantom_abstract"
        supported_spaces: ClassVar[frozenset] = frozenset()

    assert inspect.isabstract(PhantomAbstractModel)

    monkeypatch.setattr(dummy_module, "PhantomAbstractModel", PhantomAbstractModel, raising=False)

    available = get_available_models()

    assert "phantom_abstract" not in available
    assert PhantomAbstractModel not in available.values()


@pytest.mark.unit
def test_create_model_unknown_name_raises_value_error_not_type_error() -> None:
    """`create_model("tdmodel")` must fail with a clear `ValueError`, never a `TypeError`
    from instantiating an abstract class."""
    with pytest.raises(ValueError, match="tdmodel"):
        create_model("tdmodel")


@pytest.mark.unit
def test_create_model_error_names_available_models() -> None:
    with pytest.raises(ValueError) as exc_info:
        create_model("not_a_real_model")

    message = str(exc_info.value)
    for name in ("deep_q_learning", "dummy", "simple_q_learning", "simple_sarsa"):
        assert name in message

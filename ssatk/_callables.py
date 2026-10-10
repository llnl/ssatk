"""Select supported call shapes without executing a callback speculatively."""

from functools import lru_cache
import inspect


def _select_variant(model, shapes):
    try:
        signature = inspect.signature(model)
    except (TypeError, ValueError):
        # Non-inspectable callbacks use the primary documented convention.
        # Wrap one in a Python function to declare an alternate convention.
        return 0
    for index, (count, keywords) in enumerate(shapes):
        try:
            signature.bind(*([None] * count), **dict.fromkeys(keywords))
        except TypeError:
            continue
        return index
    raise TypeError("callback does not accept any supported argument signature")


_cached_variant = lru_cache(maxsize=256)(_select_variant)


def call_with_variants(model, variants):
    """Call exactly once; variants contain ``(positional_args, keyword_args)``.

    Only arity/keyword binding chooses the variant. Model-body exceptions always
    escape unchanged. Resolution is cached for hashable callbacks; unhashable
    callable instances remain supported without changing their identity.
    """
    shapes = tuple((len(args), tuple(kwargs)) for args, kwargs in variants)
    try:
        hash(model)
    except TypeError:
        index = _select_variant(model, shapes)
    else:
        index = _cached_variant(model, shapes)
    args, kwargs = variants[index]
    return model(*args, **kwargs)

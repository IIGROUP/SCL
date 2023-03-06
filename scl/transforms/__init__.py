from .mae_transforms import (
    clip_transform,
    clip_transform_randaug,
    clip_transform_test
)

_transforms = {
    "clip": clip_transform,
    "clip_randaug": clip_transform_randaug,
    "clip_test": clip_transform_test,
}


def keys_to_transforms(keys: list, size=224):
    return [_transforms[key](size=size) for key in keys]

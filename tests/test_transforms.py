from __future__ import annotations

from PIL import Image

from databaseMLUtils.transforms import dynamic_crop, resize_image


def test_dynamic_crop_square() -> None:
    img = Image.new("RGB", (100, 60), color=(0, 0, 0))
    crop = dynamic_crop(img, (10, 10, 20, 10), margin=0.0, square=True)
    assert crop.size[0] == crop.size[1]


def test_resize_image() -> None:
    img = Image.new("RGB", (20, 20), color=(0, 0, 0))
    out = resize_image(img, 16)
    assert out.size == (16, 16)


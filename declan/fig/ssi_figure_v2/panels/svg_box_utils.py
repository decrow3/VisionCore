#!/usr/bin/env python3
"""Shared SVG box-editing utilities for the layout-box export/import round
trips (panel_a_layout_boxes.py for Panel A's internal geometry,
page_layout_boxes.py for where each panel sits on the page).

The one nontrivial piece both need: resolving a `<rect id="...">`'s real
bounding box after a human has dragged/resized it in an SVG editor. Simple
attribute rewrites (x/y/width/height changed directly -- Inkscape's default)
are trivial to read, but some editors (Illustrator among them) instead wrap
the edit in a `transform="matrix(...)"` / `translate(...)` on the element
or an ancestor group, leaving x/y/width/height untouched. find_rect_bbox
composes every ancestor's transform down to the element itself so either
style of edit resolves to the same answer.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

Matrix = tuple[float, float, float, float, float, float]

IDENTITY: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def mat_compose(outer: Matrix, inner: Matrix) -> Matrix:
    a1, b1, c1, d1, e1, f1 = outer
    a2, b2, c2, d2, e2, f2 = inner
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def mat_apply(mat: Matrix, x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = mat
    return a * x + c * y + e, b * x + d * y + f


def parse_transform(transform_str: str | None) -> Matrix:
    """Compose an SVG transform= attribute into a 2x3 affine matrix.

    Only translate/scale/matrix are handled -- the only transforms a plain
    drag or corner-resize actually produces. Rotate/skew are ignored with a
    warning since a simple move/resize shouldn't create them.
    """
    mat = IDENTITY
    if not transform_str:
        return mat
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", transform_str):
        nums = [float(v) for v in re.split(r"[,\s]+", args.strip()) if v]
        if name == "translate":
            tx = nums[0]
            ty = nums[1] if len(nums) > 1 else 0.0
            mat = mat_compose(mat, (1.0, 0.0, 0.0, 1.0, tx, ty))
        elif name == "scale":
            sx = nums[0]
            sy = nums[1] if len(nums) > 1 else sx
            mat = mat_compose(mat, (sx, 0.0, 0.0, sy, 0.0, 0.0))
        elif name == "matrix" and len(nums) == 6:
            mat = mat_compose(mat, tuple(nums))
        else:
            print(f"svg_box_utils: ignoring unsupported transform '{name}(...)' (only translate/scale/matrix are applied)")
    return mat


def find_rect_bbox(root: ET.Element, box_id: str) -> tuple[float, float, float, float]:
    """Depth-first search for <rect id=box_id>, composing every ancestor's
    (and its own) transform= into the resolved axis-aligned bounding box in
    document (viewBox) units.
    """

    def walk(el: ET.Element, mat: Matrix):
        tag = el.tag.split("}")[-1]
        local_mat = mat_compose(mat, parse_transform(el.get("transform")))
        if tag == "rect" and el.get("id") == box_id:
            x = float(el.get("x", "0"))
            y = float(el.get("y", "0"))
            w = float(el.get("width", "0"))
            h = float(el.get("height", "0"))
            corners = [
                mat_apply(local_mat, x, y),
                mat_apply(local_mat, x + w, y),
                mat_apply(local_mat, x, y + h),
                mat_apply(local_mat, x + w, y + h),
            ]
            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        for child in el:
            found = walk(child, local_mat)
            if found is not None:
                return found
        return None

    result = walk(root, IDENTITY)
    if result is None:
        raise ValueError(f'svg_box_utils: no <rect id="{box_id}"> found in the edited SVG')
    return result

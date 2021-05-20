"""Utility functions for label."""
from typing import Dict, List, Tuple

from .typing import Category, Config, Label


def get_leaf_categories(parent_categories: List[Category]) -> List[Category]:
    """Get the leaf categories in the category tree structure."""
    result = []
    for category in parent_categories:
        if category.subcategories is None:
            result.append(category)
        else:
            result.extend(get_leaf_categories(category.subcategories))

    return result


def get_parent_categories(
    parent_categories: List[Category],
) -> Dict[str, List[Category]]:
    """Get all parent categories and their associated leaf categories."""
    result = {}
    for category in parent_categories:
        if category.subcategories is not None:
            result.update(get_parent_categories(category.subcategories))
            result[category.name] = get_leaf_categories([category])
        else:
            return {}
    return result


def get_category_id(category_name: str, config: Config) -> Tuple[bool, int]:
    """Get category id from category name and Config.

    We define the category id as the index (starting at 1) of the category
    within the leaf categories of the structure in MetaConfig.
    The returned boolean item means whether this instance should be ignored.
    """
    leaf_cats = get_leaf_categories(config.categories)
    leaf_cat_names = [cat.name for cat in leaf_cats]
    if category_name not in leaf_cat_names:
        return True, 0
    return False, leaf_cat_names.index(category_name) + 1


def check_crowd(label: Label) -> bool:
    """Check crowd attribute. Support for legacy behavior."""
    if label.attributes is not None:
        crowd = bool(label.attributes.get("crowd", False))
    else:
        crowd = False
    return crowd


def check_ignored(label: Label) -> bool:
    """Check ignored attribute. Support for legacy behavior."""
    if label.attributes is not None:
        ignored = bool(label.attributes.get("ignored", False))
    else:
        ignored = False
    return ignored

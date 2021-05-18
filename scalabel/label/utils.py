"""Utility functions for label."""
from typing import Dict, List

from .typing import Category, Config


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


def get_category_id(category: str, metadata_cfg: Config) -> int:
    """Get category id from category name and MetaConfig.

    We define the category id as the index (starting at 1) of the category
    within the leaf categories of the structure in MetaConfig.
    """
    leaf_cats = get_leaf_categories(metadata_cfg.categories)
    leaf_cat_names = [cat.name for cat in leaf_cats]
    if category not in leaf_cat_names:
        raise ValueError(
            f"Category {category} not in leaf categories: {leaf_cat_names}"
        )
    return leaf_cat_names.index(category) + 1

import { Key, LabelTypeName } from "../../const/common"
import { Polygon2D } from "./polygon2d"
import { Polygon2DBoundaryCloner } from "./polygon2d_boundary_cloner"
import { Label2D, Label2DModifier } from "./label2d"

/**
 * Try to create a modifier from keyboard event.
 *
 * @param activeLabels list of current active labels.
 * @param e the keyboard event.
 */
export function checkModifierFromKeyboard(
  activeLabels: Label2D[],
  e: KeyboardEvent
): Label2DModifier | undefined {
  switch (e.key) {
    case Key.D_UP:
    case Key.D_LOW:
      if (activeLabels.length === 1) {
        const target = activeLabels[0]
        if (target.type === LabelTypeName.POLYGON_2D) {
          // Pressing D with exactly one active Polygon2D label
          // trigger the Polygon2D boundary cloner modifier.
          return new Polygon2DBoundaryCloner(target as Polygon2D)
        }
      }
  }

  return undefined
}

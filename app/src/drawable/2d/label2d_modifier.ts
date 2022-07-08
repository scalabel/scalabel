import { Key, LabelTypeName } from "../../const/common"
import { Polygon2D } from "./polygon2d"
import { Polygon2DBoundaryCloner } from "./polygon2d_boundary_cloner"
import { Label2D, Label2DModifier } from "./label2d"

/**
 * Try to create a modifier from keyboard event.
 *
 * @param target
 * @param e
 */
export function checkModifierFromKeyboard(
  target: Label2D,
  e: KeyboardEvent
): Label2DModifier | undefined {
  switch (e.key) {
    case Key.D_UP:
    case Key.D_LOW:
      if (target.type === LabelTypeName.POLYGON_2D) {
        return new Polygon2DBoundaryCloner(target as Polygon2D)
      }
  }

  return undefined
}

import { Key, LabelTypeName } from "../../const/common"
import { Polygon2DBoundaryCloner } from "./polygon2d_boundary_cloner"
import { Label2D, Label2DModifier } from "./label2d"

export function checkModifierFromKeyboard(
  target: Label2D,
  e: KeyboardEvent,
): Label2DModifier | null {
  switch (e.key) {
    case Key.D_UP:
    case Key.D_LOW:
      if (target.type === LabelTypeName.POLYGON_2D) {
        return new Polygon2DBoundaryCloner(target)
      }
  }
  
  return null
}

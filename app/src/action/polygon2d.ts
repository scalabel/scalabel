import { LabelTypeName } from "../const/common"
import { makeLabel, makePathPoint2D } from "../functional/states"
import { AddLabelsAction } from "../types/action"
import { SimplePathPoint2DType } from "../types/state"
import * as actions from "./common"

/**
 * Create AddLabelAction to create a polygon2d label
 *
 * @param itemIndex
 * @param sensor
 * @param category
 * @param points list of the control points
 * @param types list of the type of the control points
 * @param closed
 * @param manual
 * @returns {AddLabelAction}
 */
export function addPolygon2dLabel(
  itemIndex: number,
  sensor: number,
  category: number[],
  points: SimplePathPoint2DType[],
  closed: boolean,
  manual = true
): AddLabelsAction {
  const labelType = closed
    ? LabelTypeName.POLYGON_2D
    : LabelTypeName.POLYLINE_2D
  const label = makeLabel({
    type: labelType,
    category,
    sensors: [sensor],
    manual
  })
  const shapes = points.map((p) => makePathPoint2D({ ...p, label: [label.id] }))
  label.shapes = shapes.map((s) => s.id)
  return actions.addLabel(itemIndex, label, shapes)
}

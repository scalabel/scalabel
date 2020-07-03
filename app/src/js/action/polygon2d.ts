import { LabelTypeName } from '../common/types'
import { makeLabel, makePathPoint2D } from '../functional/states'
import { SimplePathPoint2DType } from '../functional/types'
import * as actions from './common'
import { AddLabelsAction } from './types'

/**
 * Create AddLabelAction to create a polygon2d label
 * @param itemIndex
 * @param category
 * @param points list of the control points
 * @param types list of the type of the control points
 * @return {AddLabelAction}
 */
export function addPolygon2dLabel (
  itemIndex: number,
  sensor: number,
  category: number[],
  points: SimplePathPoint2DType[],
  closed: boolean,
  manual = true
)
  : AddLabelsAction {
  const labelType = closed ?
    LabelTypeName.POLYGON_2D : LabelTypeName.POLYLINE_2D
  const label = makeLabel({
    type: labelType, category, sensors: [sensor], manual
  })
  const shapes = points.map((p) => makePathPoint2D({ ...p, label: [label.id] }))
  label.shapes = shapes.map((s) => s.id)
  return actions.addLabel(itemIndex, label, shapes)
}

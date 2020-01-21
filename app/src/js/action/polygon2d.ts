import { LabelTypeName, ShapeTypeName } from '../common/types'
import { makeLabel } from '../functional/states'
import { PathPoint2DType } from '../functional/types'
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
  points: PathPoint2DType[],
  closed: boolean
)
  : AddLabelsAction {
  const labelType = closed ?
    LabelTypeName.POLYGON_2D : LabelTypeName.POLYLINE_2D
  const label = makeLabel({
    type: labelType, category, sensors: [sensor]
  })
  return actions.addLabel(
    itemIndex, label, [ShapeTypeName.POLYGON_2D], points
  )
}

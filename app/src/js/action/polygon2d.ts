import { LabelTypeName } from '../common/types'
import { makeLabel, makePolygon } from '../functional/states'
import { PolyPathPoint2DType } from '../functional/types'
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
  points: PolyPathPoint2DType[],
  closed: boolean,
  manual = true
)
  : AddLabelsAction {
  const polygon = makePolygon({ points })
  const labelType = closed ?
    LabelTypeName.POLYGON_2D : LabelTypeName.POLYLINE_2D
  const label = makeLabel({
    type: labelType, category, sensors: [sensor], manual
  })
  return actions.addLabel(
    itemIndex, label, [polygon]
  )
}

import { LabelTypeName } from '../const/common'
import { makeLabel, makePlane } from '../functional/states'
import { AddLabelsAction } from '../types/action'
import { Vector3Type } from '../types/functional'
import * as actions from './common'

/**
 * Create AddLabelAction to create a box3d label
 * @param {number} itemIndex
 * @param {number[]} category: list of category ids
 * @param {number} center
 * @param {number} size
 * @param {number} orientation
 * @return {AddLabelAction}
 */
export function addPlaneLabel (
  itemIndex: number,
  sensors: number[],
  offset: Vector3Type,
  orientation: Vector3Type
): AddLabelsAction {
  // Create the rect object
  const plane = makePlane({ offset, orientation })
  const label = makeLabel({ type: LabelTypeName.PLANE_3D, sensors })
  return actions.addLabel(
    itemIndex, label, [plane]
  )
}

import { LabelTypes } from '../common/types'
import { makeLabel, makePlane } from '../functional/states'
import { Vector3Type } from '../functional/types'
import * as actions from './common'
import { AddLabelsAction } from './types'

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
  offset: Vector3Type,
  orientation: Vector3Type): AddLabelsAction {
  // create the rect object
  const plane = makePlane({ offset, orientation })
  const label = makeLabel({ type: LabelTypes.PLANE_3D })
  return actions.addLabel(itemIndex, label, [LabelTypes.PLANE_3D], [plane])
}

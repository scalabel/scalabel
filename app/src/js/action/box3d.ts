import * as labels from '../common/label_types'
import { makeCube, makeLabel } from '../functional/states'
import { Vector3Type } from '../functional/types'
import * as actions from './common'
import { AddLabelAction } from './types'

/**
 * Create AddLabelAction to create a box3d label
 * @param {number} itemIndex
 * @param {number[]} category: list of category ids
 * @param {number} center
 * @param {number} size
 * @param {number} orientation
 * @return {AddLabelAction}
 */
export function addBox3dLabel (
  itemIndex: number, category: number[],
  center: Vector3Type, size: Vector3Type,
  orientation: Vector3Type): AddLabelAction {
  // create the rect object
  const cube = makeCube({ center, size, orientation })
  const label = makeLabel({ type: labels.BOX_3D, category })
  return actions.addLabel(itemIndex, label, [cube])
}

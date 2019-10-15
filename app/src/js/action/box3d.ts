import * as types from '../common/types'
import { makeCube, makeLabel } from '../functional/states'
import { Vector3Type } from '../functional/types'
import { addLabel } from './common'
import { BaseAction } from './types'

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
  orientation: Vector3Type, surfaceId: number = -1): BaseAction {
  // create the rect object
  const cube = makeCube({ center, size, orientation, surfaceId })
  const label = makeLabel({ type: types.LabelTypeName.BOX_3D, category })

  return addLabel(itemIndex, label, [types.ShapeType.CUBE], [cube])
}

import * as types from "../const/common"
import { makeCube, makeLabel } from "../functional/states"
import { AddLabelsAction } from "../types/action"
import { Vector3Type } from "../types/state"
import { addLabel } from "./common"

/**
 * Create AddLabelAction to create a box3d label
 *
 * @param {number} itemIndex
 * @param {number[]} category: list of category ids
 * @param sensors
 * @param category
 * @param {number} center
 * @param {number} size
 * @param {number} orientation
 * @returns {AddLabelAction}
 */
export function addBox3dLabel(
  itemIndex: number,
  sensors: number[],
  category: number[],
  center: Vector3Type,
  size: Vector3Type,
  orientation: Vector3Type
): AddLabelsAction {
  // Create the rect object
  const cube = makeCube({ center, size, orientation })
  const label = makeLabel({
    type: types.LabelTypeName.BOX_3D,
    category,
    sensors
  })

  return addLabel(itemIndex, label, [cube])
}

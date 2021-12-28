import { Box3D } from "../drawable/3d/box3d"
import { Label3DList } from "../drawable/3d/label3d_list"
import { commitLabels } from "../drawable/states"
import { Vector3D } from "../math/vector3d"

/**
 * Create AddLabelAction to create a box3d label
 *
 * @param labelList
 * @param itemIndex
 * @param sensors
 * @param category
 * @param center
 * @param dimension
 * @param orientation
 * @param tracking
 */
export function addBox3dLabel(
  labelList: Label3DList,
  itemIndex: number,
  sensors: number[],
  category: number,
  center: Vector3D,
  dimension: Vector3D,
  orientation: Vector3D,
  tracking: boolean
): void {
  const box = new Box3D(labelList)
  box.init(itemIndex, category, center, orientation, dimension, sensors)
  commitLabels([box], tracking)
}

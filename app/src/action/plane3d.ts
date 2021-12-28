import { Label3DList } from "../drawable/3d/label3d_list"
import { Plane3D } from "../drawable/3d/plane3d"
import { commitLabels } from "../drawable/states"
import { Vector3D } from "../math/vector3d"

/**
 * Commit new plane 3D label to state
 *
 * @param labelList
 * @param itemIndex
 * @param category
 * @param sensors
 * @param center
 * @param orientation
 */
export function addPlaneLabel(
  labelList: Label3DList,
  itemIndex: number,
  category: number,
  center?: Vector3D,
  orientation?: Vector3D,
  sensors?: number[]
): void {
  const plane = new Plane3D(labelList)
  plane.init(itemIndex, category, center, orientation, sensors)
  commitLabels([plane], false)
}

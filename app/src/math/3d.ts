import * as THREE from "three"

/**
 * Get axis aligned scale in coordinate frame rotated by quaternion
 *
 * @param scale
 * @param quaternion
 */
export function rotateScale(
  scale: THREE.Vector3,
  quaternion: THREE.Quaternion
): THREE.Vector3 {
  const rotatedScaleArray = [1, 1, 1]

  for (let i = 0; i < 3; i++) {
    const axisArray = [0, 0, 0]
    axisArray[i] = 1
    const axis = new THREE.Vector3()
    axis.fromArray(axisArray)
    axis.applyQuaternion(quaternion)
    axis.multiply(scale)
    rotatedScaleArray[i] = axis.length()
  }

  const rotatedScale = new THREE.Vector3()
  rotatedScale.fromArray(rotatedScaleArray)

  return rotatedScale
}

import * as THREE from 'three'
import Session from '../common/session'
import { PointCloudViewerConfigType } from '../functional/types'
import { Vector3D } from '../math/vector3d'
import * as types from './types'

export const MOUSE_CORRECTION_FACTOR = 60.0
export const MOVE_AMOUNT = 0.3
export const ZOOM_SPEED = 1.1

/**
 * Generate move camera action
 * @param {Vector3D} newPosition
 */
export function moveCamera (
  newPosition: Vector3D): types.UpdatePointCloudViewerConfigAction {
  return {
    type: types.UPDATE_POINT_CLOUD_VIEWER_CONFIG,
    sessionId: Session.id,
    newFields: {
      position: newPosition.toObject()
    }
  }
}

/**
 * Generate move camera action and target
 * @param {Vector3D} newPosition
 * @param {Vector3D} newTarget
 */
export function moveCameraAndTarget (
  newPosition: Vector3D,
  newTarget: Vector3D
): types.UpdatePointCloudViewerConfigAction {
  return {
    type: types.UPDATE_POINT_CLOUD_VIEWER_CONFIG,
    sessionId: Session.id,
    newFields: {
      position: newPosition.toObject(),
      target: newTarget.toObject()
    }
  }
}

/**
 * Zoom camera
 */
export function zoomCamera (
  viewerConfig: PointCloudViewerConfigType,
  deltaY: number
): types.UpdatePointCloudViewerConfigAction | null {
  const target = new THREE.Vector3(viewerConfig.target.x,
      viewerConfig.target.y,
      viewerConfig.target.z)
  const offset = new THREE.Vector3(viewerConfig.position.x,
      viewerConfig.position.y,
      viewerConfig.position.z)
  offset.sub(target)

  const spherical = new THREE.Spherical()
  spherical.setFromVector3(offset)

    // Decrease distance from origin by amount specified
  let newRadius = spherical.radius
  if (deltaY > 0) {
    newRadius *= ZOOM_SPEED
  } else {
    newRadius /= ZOOM_SPEED
  }
    // Limit zoom to not be too close
  if (newRadius > 0.1 && newRadius < 500) {
    spherical.radius = newRadius

    offset.setFromSpherical(spherical)

    offset.add(target)

    return moveCamera(new Vector3D().fromThree(offset))
  }
  return null
}

/**
 * Rotate camera according to mouse movement
 * @param newX
 * @param newY
 */
export function rotateCamera (
  initialX: number,
  initialY: number,
  newX: number,
  newY: number,
  viewerConfig: PointCloudViewerConfigType
): types.UpdatePointCloudViewerConfigAction {
  const target = new THREE.Vector3(viewerConfig.target.x,
    viewerConfig.target.y,
    viewerConfig.target.z)
  const offset = new THREE.Vector3(viewerConfig.position.x,
    viewerConfig.position.y,
    viewerConfig.position.z)
  offset.sub(target)

  // Rotate so that positive y-axis is vertical
  const rotVertQuat = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(viewerConfig.verticalAxis.x,
      viewerConfig.verticalAxis.y,
      viewerConfig.verticalAxis.z),
    new THREE.Vector3(0, 1, 0))
  offset.applyQuaternion(rotVertQuat)

  // Convert to spherical coordinates
  const spherical = new THREE.Spherical()
  spherical.setFromVector3(offset)

  // Apply rotations
  spherical.theta += (newX - initialX) / MOUSE_CORRECTION_FACTOR
  spherical.phi += (newY - initialY) / MOUSE_CORRECTION_FACTOR

  spherical.phi = Math.max(0, Math.min(Math.PI, spherical.phi))

  spherical.makeSafe()

  // Convert to Cartesian
  offset.setFromSpherical(spherical)

  // Rotate back to original coordinate space
  const quatInverse = rotVertQuat.clone().inverse()
  offset.applyQuaternion(quatInverse)

  offset.add(target)

  return moveCamera((new Vector3D()).fromThree(offset))
}

/**
 * Drag camera according to mouse movement
 * @param newX
 * @param newY
 */
export function dragCamera (
  initialX: number,
  initialY: number,
  newX: number,
  newY: number,
  camera: THREE.Camera,
  viewerConfig: PointCloudViewerConfigType
): types.UpdatePointCloudViewerConfigAction {
  const dragVector = new THREE.Vector3(
    (initialX - newX) / MOUSE_CORRECTION_FACTOR * 2,
    (newY - initialY) / MOUSE_CORRECTION_FACTOR * 2,
    0
  )
  dragVector.applyQuaternion(camera.quaternion)

  return (moveCameraAndTarget(
    new Vector3D(
      viewerConfig.position.x + dragVector.x,
      viewerConfig.position.y + dragVector.y,
      viewerConfig.position.z + dragVector.z
    ),
    new Vector3D(
      viewerConfig.target.x + dragVector.x,
      viewerConfig.target.y + dragVector.y,
      viewerConfig.target.z + dragVector.z
    )
  ))
}

/**
 * Move camera up
 * @param viewerConfig
 */
export function moveUp (
  viewerConfig: PointCloudViewerConfigType
): types.UpdatePointCloudViewerConfigAction {
  return moveCameraAndTarget(
    new Vector3D(
      viewerConfig.position.x,
      viewerConfig.position.y,
      viewerConfig.position.z + MOVE_AMOUNT
    ),
    new Vector3D(
      viewerConfig.target.x,
      viewerConfig.target.y,
      viewerConfig.target.z + MOVE_AMOUNT
    )
  )
}

/**
 * Move camera up
 * @param viewerConfig
 */
export function moveDown (
  viewerConfig: PointCloudViewerConfigType
): types.UpdatePointCloudViewerConfigAction {
  return moveCameraAndTarget(
    new Vector3D(
      viewerConfig.position.x,
      viewerConfig.position.y,
      viewerConfig.position.z - MOVE_AMOUNT
    ),
    new Vector3D(
      viewerConfig.target.x,
      viewerConfig.target.y,
      viewerConfig.target.z - MOVE_AMOUNT
    )
  )
}

/**
 * Calculate forward vector
 * @param viewerConfig
 */
function calculateForward (
  viewerConfig: PointCloudViewerConfigType
): THREE.Vector3 {
  // Get vector pointing from camera to target projected to horizontal plane
  let forwardX = viewerConfig.target.x - viewerConfig.position.x
  let forwardY = viewerConfig.target.y - viewerConfig.position.y
  const forwardDist = Math.sqrt(forwardX * forwardX + forwardY * forwardY)
  forwardX *= MOVE_AMOUNT / forwardDist
  forwardY *= MOVE_AMOUNT / forwardDist
  return new THREE.Vector3(forwardX, forwardY, 0)
}

/**
 * Move camera back
 * @param viewerConfig
 */
export function moveBack (
  viewerConfig: PointCloudViewerConfigType
): types.UpdatePointCloudViewerConfigAction {
  const forward = calculateForward(viewerConfig)
  return moveCameraAndTarget(
    new Vector3D(
      viewerConfig.position.x - forward.x,
      viewerConfig.position.y - forward.y,
      viewerConfig.position.z
    ),
    new Vector3D(
      viewerConfig.target.x - forward.x,
      viewerConfig.target.y - forward.y,
      viewerConfig.target.z
    )
  )
}

/**
 * Move camera back
 * @param viewerConfig
 */
export function moveForward (
  viewerConfig: PointCloudViewerConfigType
): types.UpdatePointCloudViewerConfigAction {
  const forward = calculateForward(viewerConfig)
  return moveCameraAndTarget(
    new Vector3D(
      viewerConfig.position.x + forward.x,
      viewerConfig.position.y + forward.y,
      viewerConfig.position.z
    ),
    new Vector3D(
      viewerConfig.target.x + forward.x,
      viewerConfig.target.y + forward.y,
      viewerConfig.target.z
    )
  )
}

/**
 * Calculate left vector
 * @param viewerConfig
 * @param forward
 */
function calculateLeft (
  viewerConfig: PointCloudViewerConfigType,
  forward: THREE.Vector3
): THREE.Vector3 {
  // Get vector pointing up
  const vertical = new THREE.Vector3(
    viewerConfig.verticalAxis.x,
    viewerConfig.verticalAxis.y,
    viewerConfig.verticalAxis.z
  )

  // Handle movement in three dimensions
  const left = new THREE.Vector3()
  left.crossVectors(vertical, forward)
  left.normalize()
  left.multiplyScalar(MOVE_AMOUNT)
  return left
}

/**
 * Move camera left
 * @param viewerConfig
 */
export function moveLeft (
  viewerConfig: PointCloudViewerConfigType
): types.UpdatePointCloudViewerConfigAction {
  const forward = calculateForward(viewerConfig)
  const left = calculateLeft(viewerConfig, forward)
  return moveCameraAndTarget(
    new Vector3D(
      viewerConfig.position.x + left.x,
      viewerConfig.position.y + left.y,
      viewerConfig.position.z + left.z
    ),
    new Vector3D(
      viewerConfig.target.x + left.x,
      viewerConfig.target.y + left.y,
      viewerConfig.target.z + left.z
    )
  )
}

/**
 * Move camera right
 * @param viewerConfig
 */
export function moveRight (
  viewerConfig: PointCloudViewerConfigType
): types.UpdatePointCloudViewerConfigAction {
  const forward = calculateForward(viewerConfig)
  const left = calculateLeft(viewerConfig, forward)
  return moveCameraAndTarget(
    new Vector3D(
      viewerConfig.position.x - left.x,
      viewerConfig.position.y - left.y,
      viewerConfig.position.z - left.z
    ),
    new Vector3D(
      viewerConfig.target.x - left.x,
      viewerConfig.target.y - left.y,
      viewerConfig.target.z - left.z
    )
  )
}

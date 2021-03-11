import * as THREE from "three"

import * as actionConsts from "../const/action"
import { Vector3D } from "../math/vector3d"
import * as actionTypes from "../types/action"
import { PointCloudViewerConfigType, ViewerConfigType } from "../types/state"
import { changeViewerConfig, makeBaseAction } from "./common"

export enum CameraMovementParameters {
  MOUSE_CORRECTION_FACTOR = 60.0,
  MOVE_AMOUNT = 0.15,
  ZOOM_SPEED = 1.05
}

export enum CameraLockState {
  UNLOCKED = 0,
  X_LOCKED = 1,
  Y_LOCKED = 2,
  SELECTION_X = 3,
  SELECTION_Y = 4,
  SELECTION_Z = 5
}

/**
 * Returns whether camera lock state is locked to selected label
 *
 * @param viewerConfig
 */
export function lockedToSelection(
  viewerConfig: PointCloudViewerConfigType
): boolean {
  return viewerConfig.lockStatus >= CameraLockState.SELECTION_X
}

/**
 * Generate move camera action
 *
 * @param {Vector3D} newPosition
 * @param viewerId
 * @param viewerConfig
 */
export function moveCamera(
  newPosition: Vector3D,
  viewerId: number,
  viewerConfig: ViewerConfigType
): actionTypes.ChangeViewerConfigAction {
  const config = {
    ...viewerConfig,
    position: newPosition.toState()
  }
  return {
    ...makeBaseAction(actionConsts.CHANGE_VIEWER_CONFIG),
    viewerId,
    config
  }
}

/**
 * Generate move camera action and target
 *
 * @param {Vector3D} newPosition
 * @param {Vector3D} newTarget
 * @param viewerId
 * @param viewerConfig
 */
export function moveCameraAndTarget(
  newPosition: Vector3D,
  newTarget: Vector3D,
  viewerId: number,
  viewerConfig: ViewerConfigType
): actionTypes.ChangeViewerConfigAction {
  const config = {
    ...viewerConfig,
    position: newPosition.toState(),
    target: newTarget.toState()
  }
  return {
    ...makeBaseAction(actionConsts.CHANGE_VIEWER_CONFIG),
    viewerId,
    config
  }
}

/**
 * Zoom camera
 *
 * @param deltaY
 * @param viewerId
 * @param viewerConfig
 */
export function zoomCamera(
  deltaY: number,
  viewerId: number,
  viewerConfig: PointCloudViewerConfigType
): actionTypes.ChangeViewerConfigAction | null {
  const target = new THREE.Vector3(
    viewerConfig.target.x,
    viewerConfig.target.y,
    viewerConfig.target.z
  )
  const offset = new THREE.Vector3(
    viewerConfig.position.x,
    viewerConfig.position.y,
    viewerConfig.position.z
  )
  offset.sub(target)

  const spherical = new THREE.Spherical()
  spherical.setFromVector3(offset)

  // Decrease distance from origin by amount specified
  let newRadius = spherical.radius
  if (deltaY > 0) {
    newRadius *= CameraMovementParameters.ZOOM_SPEED
  } else {
    newRadius /= CameraMovementParameters.ZOOM_SPEED
  }
  // Limit zoom to not be too close
  if (newRadius > 0.1 && newRadius < 500) {
    spherical.radius = newRadius

    offset.setFromSpherical(spherical)

    offset.add(target)

    return moveCamera(new Vector3D().fromThree(offset), viewerId, viewerConfig)
  }
  return null
}

/**
 * Rotate camera according to mouse movement
 *
 * @param initialX
 * @param initialY
 * @param newX
 * @param newY
 * @param viewerId
 * @param viewerConfig
 */
export function rotateCamera(
  initialX: number,
  initialY: number,
  newX: number,
  newY: number,
  viewerId: number,
  viewerConfig: PointCloudViewerConfigType
): actionTypes.ChangeViewerConfigAction {
  if (viewerConfig.lockStatus === CameraLockState.X_LOCKED) {
    newX = initialX
  } else if (viewerConfig.lockStatus === CameraLockState.Y_LOCKED) {
    newY = initialY
  }
  const target = new THREE.Vector3(
    viewerConfig.target.x,
    viewerConfig.target.y,
    viewerConfig.target.z
  )
  const offset = new THREE.Vector3(
    viewerConfig.position.x,
    viewerConfig.position.y,
    viewerConfig.position.z
  )

  offset.sub(target)

  // Rotate so that positive y-axis is vertical
  const rotVertQuat = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(
      viewerConfig.verticalAxis.x,
      viewerConfig.verticalAxis.y,
      viewerConfig.verticalAxis.z
    ),
    new THREE.Vector3(0, 1, 0)
  )
  offset.applyQuaternion(rotVertQuat)

  // Convert to spherical coordinates
  const spherical = new THREE.Spherical()
  spherical.setFromVector3(offset)

  // Apply rotations
  spherical.theta +=
    (newX - initialX) / CameraMovementParameters.MOUSE_CORRECTION_FACTOR
  spherical.phi +=
    (newY - initialY) / CameraMovementParameters.MOUSE_CORRECTION_FACTOR

  spherical.phi = Math.max(0, Math.min(Math.PI, spherical.phi))

  spherical.makeSafe()

  // Convert to Cartesian
  offset.setFromSpherical(spherical)

  // Rotate back to original coordinate space
  const quatInverse = rotVertQuat.clone().inverse()
  offset.applyQuaternion(quatInverse)
  offset.add(target)

  return moveCamera(new Vector3D().fromThree(offset), viewerId, viewerConfig)
}

/**
 * Drag camera according to mouse movement
 *
 * @param initialX
 * @param initialY
 * @param newX
 * @param newY
 * @param camera
 * @param viewerId
 * @param viewerConfig
 */
export function dragCamera(
  initialX: number,
  initialY: number,
  newX: number,
  newY: number,
  camera: THREE.Camera,
  viewerId: number,
  viewerConfig: PointCloudViewerConfigType
): actionTypes.ChangeViewerConfigAction {
  const dragVector = new THREE.Vector3(
    ((initialX - newX) / CameraMovementParameters.MOUSE_CORRECTION_FACTOR) * 2,
    ((newY - initialY) / CameraMovementParameters.MOUSE_CORRECTION_FACTOR) * 2,
    0
  )
  dragVector.applyQuaternion(camera.quaternion)

  return moveCameraAndTarget(
    new Vector3D(
      viewerConfig.position.x + dragVector.x,
      viewerConfig.position.y + dragVector.y,
      viewerConfig.position.z + dragVector.z
    ),
    new Vector3D(
      viewerConfig.target.x + dragVector.x,
      viewerConfig.target.y + dragVector.y,
      viewerConfig.target.z + dragVector.z
    ),
    viewerId,
    viewerConfig
  )
}

/**
 * Move camera up
 *
 * @param viewerId
 * @param viewerConfig
 */
export function moveUp(
  viewerId: number,
  viewerConfig: PointCloudViewerConfigType
): actionTypes.ChangeViewerConfigAction {
  return moveCameraAndTarget(
    new Vector3D(
      viewerConfig.position.x,
      viewerConfig.position.y,
      viewerConfig.position.z + CameraMovementParameters.MOVE_AMOUNT
    ),
    new Vector3D(
      viewerConfig.target.x,
      viewerConfig.target.y,
      viewerConfig.target.z + CameraMovementParameters.MOVE_AMOUNT
    ),
    viewerId,
    viewerConfig
  )
}

/**
 * Move camera up
 *
 * @param viewerId
 * @param viewerConfig
 */
export function moveDown(
  viewerId: number,
  viewerConfig: PointCloudViewerConfigType
): actionTypes.ChangeViewerConfigAction {
  return moveCameraAndTarget(
    new Vector3D(
      viewerConfig.position.x,
      viewerConfig.position.y,
      viewerConfig.position.z - CameraMovementParameters.MOVE_AMOUNT
    ),
    new Vector3D(
      viewerConfig.target.x,
      viewerConfig.target.y,
      viewerConfig.target.z - CameraMovementParameters.MOVE_AMOUNT
    ),
    viewerId,
    viewerConfig
  )
}

/**
 * Calculate forward vector
 *
 * @param viewerConfig
 */
function calculateForward(
  viewerConfig: PointCloudViewerConfigType
): THREE.Vector3 {
  // Get vector pointing from camera to target projected to horizontal plane
  let forwardX = viewerConfig.target.x - viewerConfig.position.x
  let forwardY = viewerConfig.target.y - viewerConfig.position.y
  const forwardDist = Math.sqrt(forwardX * forwardX + forwardY * forwardY)
  forwardX *= CameraMovementParameters.MOVE_AMOUNT / forwardDist
  forwardY *= CameraMovementParameters.MOVE_AMOUNT / forwardDist
  return new THREE.Vector3(forwardX, forwardY, 0)
}

/**
 * Move camera back
 *
 * @param viewerId
 * @param viewerConfig
 */
export function moveBack(
  viewerId: number,
  viewerConfig: PointCloudViewerConfigType
): actionTypes.ChangeViewerConfigAction {
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
    ),
    viewerId,
    viewerConfig
  )
}

/**
 * Move camera back
 *
 * @param viewerId
 * @param viewerConfig
 */
export function moveForward(
  viewerId: number,
  viewerConfig: PointCloudViewerConfigType
): actionTypes.ChangeViewerConfigAction {
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
    ),
    viewerId,
    viewerConfig
  )
}

/**
 * Calculate left vector
 *
 * @param viewerConfig
 * @param forward
 */
function calculateLeft(
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
  left.multiplyScalar(CameraMovementParameters.MOVE_AMOUNT)
  return left
}

/**
 * Move camera left
 *
 * @param viewerId
 * @param viewerConfig
 */
export function moveLeft(
  viewerId: number,
  viewerConfig: PointCloudViewerConfigType
): actionTypes.ChangeViewerConfigAction {
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
    ),
    viewerId,
    viewerConfig
  )
}

/**
 * Move camera right
 *
 * @param viewerId
 * @param viewerConfig
 */
export function moveRight(
  viewerId: number,
  viewerConfig: PointCloudViewerConfigType
): actionTypes.ChangeViewerConfigAction {
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
    ),
    viewerId,
    viewerConfig
  )
}

/**
 * Update lockStatus in viewerConfig
 *
 * @param viewerId
 * @param viewerConfig
 * @param newLockStatus
 */
export function updateLockStatus(
  viewerId: number,
  viewerConfig: PointCloudViewerConfigType,
  newLockStatus: number
): actionTypes.ChangeViewerConfigAction {
  const config = {
    ...viewerConfig,
    lockStatus: newLockStatus
  }
  return changeViewerConfig(viewerId, config)
}

/**
 * Convert axis (0, 1, 2) to the selection lock state
 *
 * @param axis
 */
function axisIndexToSelectionLockState(axis: number): number {
  return CameraLockState.SELECTION_X + axis
}

/**
 * Align camera to axis
 *
 * @param viewerId
 * @param viewerConfig
 * @param axis
 * @param minDistance
 */
export function alignToAxis(
  viewerId: number,
  viewerConfig: PointCloudViewerConfigType,
  axis: number,
  minDistance: number = 3
): actionTypes.ChangeViewerConfigAction {
  const config = { ...viewerConfig }
  if (lockedToSelection(viewerConfig)) {
    config.lockStatus = axisIndexToSelectionLockState(axis)
  } else {
    const position = new Vector3D().fromState(viewerConfig.position)
    const target = new Vector3D().fromState(viewerConfig.target)
    for (let i = 0; i < 3; i++) {
      if (i !== axis) {
        position[i] = target[i] + 0.01
      } else if (Math.abs(position[i] - target[i]) < minDistance) {
        let sign = Math.sign(position[i] - target[i])
        if (sign === 0) {
          sign = 1
        }
        position[i] = sign * minDistance + target[i]
      }
    }

    config.position = position.toState()
  }

  return changeViewerConfig(viewerId, config)
}

/**
 * Lock selection to camera
 *
 * @param viewerId
 * @param viewerConfig
 */
export function toggleSelectionLock(
  viewerId: number,
  viewerConfig: PointCloudViewerConfigType
): actionTypes.ChangeViewerConfigAction {
  const config = { ...viewerConfig }
  if (lockedToSelection(viewerConfig)) {
    config.lockStatus = CameraLockState.UNLOCKED
  } else {
    config.lockStatus = CameraLockState.SELECTION_X
  }

  return changeViewerConfig(viewerId, config)
}

/**
 * Toggle drag rotation direction
 *
 * @param viewerId
 * @param viewerConfig
 */
export function toggleRotation(
  viewerId: number,
  viewerConfig: PointCloudViewerConfigType
): actionTypes.ChangeViewerConfigAction {
  // Spread the original config
  const config = { ...viewerConfig }
  if (viewerConfig.cameraRotateDir !== undefined) {
    config.cameraRotateDir = false
  } else {
    config.cameraRotateDir = true
  }

  return changeViewerConfig(viewerId, config)
}

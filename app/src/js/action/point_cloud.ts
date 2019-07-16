import Session from '../common/session'
import { Vector3D } from '../math/vector3d'
import * as types from './types'

/**
 * Generate move camera action
 * @param {Vector3D} newPosition
 */
export function moveCamera (
  newPosition: Vector3D): types.MoveCameraAndTargetAction {
  return {
    type: types.MOVE_CAMERA_AND_TARGET,
    sessionId: Session.id,
    newPosition: newPosition.toObject()
  }
}

/**
 * Generate move camera action and target
 * @param {Vector3D} newPosition
 * @param {Vector3D} newTarget
 */
export function moveCameraAndTarget (
  newPosition: Vector3D, newTarget: Vector3D): types.MoveCameraAndTargetAction {
  return {
    type: types.MOVE_CAMERA_AND_TARGET,
    sessionId: Session.id,
    newPosition: newPosition.toObject(),
    newTarget: newTarget.toObject()
  }
}

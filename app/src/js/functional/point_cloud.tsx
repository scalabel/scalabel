import {getCurrentItemViewerConfig,
  setCurrentItemViewerConfig} from './state_util';
import {updateObject} from './util';
import {State, Vector3Type} from './types';

/**
 * Move camera position to new position
 * @param {State} state: Current state
 * @param {Object} newPosition: New camera position (x, y, z)
 * @return {State}
 */
export function moveCamera(state: State,
                           newPosition: Vector3Type): State {
  let config = getCurrentItemViewerConfig(state);
  config = updateObject(config, {position: newPosition});
  return setCurrentItemViewerConfig(state, config);
}

/**
 * Move camera and target position
 * @param {State} state: Current state
 * @param {Object} newPosition: New camera position (x, y, z)
 * @param {Object} newTarget: New target position (x, y, z)
 * @return {State}
 */
export function moveCameraAndTarget(state: State, newPosition: Vector3Type,
                                    newTarget: Vector3Type): State {
  let config = getCurrentItemViewerConfig(state);
  config = updateObject(config, {position: newPosition, target: newTarget});
  return setCurrentItemViewerConfig(state, config);
}

import _ from 'lodash'
import * as types from '../action/types'
import {getCurrentItemViewerConfig,
  setCurrentItemViewerConfig} from './state_util'
import { State } from './types'
import { updateObject } from './util'

export enum EditMode {
  MOVE,
  SCALE,
  EXTRUDE,
  ROTATE
}

/**
 * Move camera and target position
 * @param {State} state: Current state
 * @param {action: types.MoveCameraAndTargetAction} action
 * @return {State}
 */
export function moveCameraAndTarget (
  state: State, action: types.MoveCameraAndTargetAction): State {
  let config = getCurrentItemViewerConfig(state)
  if ('newTarget' in action) {
    config = updateObject(
      config, { position: action.newPosition, target: action.newTarget })
  } else {
    config = updateObject(config, { position: action.newPosition })
  }
  return setCurrentItemViewerConfig(state, config)
}

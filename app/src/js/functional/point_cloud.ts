import _ from 'lodash'
import * as types from '../action/types'
import {getCurrentPointCloudViewerConfig,
  setCurrentPointCloudViewerConfig} from './state_util'
import { State } from './types'
import { updateObject } from './util'

/**
 * Move camera and target position
 * @param {State} state: Current state
 * @param {action: types.MoveCameraAndTargetAction} action
 * @return {State}
 */
export function moveCameraAndTarget (
  state: State, action: types.UpdatePointCloudViewerConfigAction): State {
  let config = getCurrentPointCloudViewerConfig(state)
  config = updateObject(
    config, action.newFields)
  return setCurrentPointCloudViewerConfig(state, config)
}

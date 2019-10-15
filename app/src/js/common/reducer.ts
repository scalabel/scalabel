import { AnyAction, Reducer } from 'redux'
import * as types from '../action/types'
import * as common from '../functional/common'
import * as image from '../functional/image'
import * as pointCloud from '../functional/point_cloud'
import { makeState } from '../functional/states'
import { State } from '../functional/types'

/**
 * Reducer
 * @param {State} currentState
 * @param {AnyAction} action
 * @return {State}
 */
export const reducer: Reducer<State> = (
    currentState: State = makeState(),
    action: AnyAction): State => {
  // Appending actions to action array
  // const newActions = currentState.actions.slice();
  // newActions.push(action);
  // const state = {...currentState, actions: newActions};
  const state = currentState
  // Apply reducers to state
  switch (action.type) {
    case types.INIT_SESSION:
      return common.initSession(state)
    case types.CHANGE_SELECT:
      return common.changeSelect(state, action as types.ChangeSelectAction)
    case types.LOAD_ITEM:
      return common.loadItem(state, action as types.LoadItemAction)
    case types.UPDATE_ALL:
      return common.updateAll(state)
    case types.UPDATE_TASK:
      return common.updateTask(state, action as types.UpdateTaskAction)
    case types.UPDATE_IMAGE_VIEWER_CONFIG:
      return image.updateImageViewerConfig(
        state, action as types.UpdateImageViewerConfigAction
      )
    case types.ADD_LABELS:
      return common.addLabels(state, action as types.AddLabelsAction)
    case types.ADD_TRACK:
      return common.addTrack(state, action as types.AddTrackAction)
    case types.CHANGE_SHAPES:
      return common.changeShapes(state, action as types.ChangeShapesAction)
    case types.CHANGE_LABELS:
      return common.changeLabels(
        state, action as types.ChangeLabelsAction)
    case types.LINK_LABELS:
      return common.linkLabels(state, action as types.LinkLabelsAction)
    case types.DELETE_LABELS:
      return common.deleteLabels(state, action as types.DeleteLabelsAction)
    case types.TOGGLE_ASSISTANT_VIEW:
      return common.toggleAssistantView(state)
    case types.UPDATE_POINT_CLOUD_VIEWER_CONFIG:
      return pointCloud.moveCameraAndTarget(
        state, action as types.UpdatePointCloudViewerConfigAction)
    default:
  }
  return state
}

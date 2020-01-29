import { AnyAction, Reducer } from 'redux'
import * as types from '../action/types'
import * as common from '../functional/common'
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
    case types.UNLINK_LABELS:
      return common.unlinkLabels(state, action as types.UnlinkLabelsAction)
    case types.MERGE_TRACKS:
      return common.mergeTracks(state, action as types.MergeTrackAction)
    case types.DELETE_LABELS:
      return common.deleteLabels(state, action as types.DeleteLabelsAction)
    case types.ADD_VIEWER_CONFIG:
      return common.addViewerConfig(
        state, action as types.AddViewerConfigAction
      )
    case types.UPDATE_PANE:
      return common.updatePane(state, action as types.UpdatePaneAction)
    case types.SPLIT_PANE:
      return common.splitPane(state, action as types.SplitPaneAction)
    case types.DELETE_PANE:
      return common.deletePane(state, action as types.DeletePaneAction)
    case types.CHANGE_VIEWER_CONFIG:
      return common.changeViewerConfig(
        state, action as types.ChangeViewerConfigAction
      )
    default:
  }
  return state
}

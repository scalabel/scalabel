import { AnyAction, Reducer } from 'redux'
import * as types from '../action/types'
import * as common from '../functional/common'
import { makeState } from '../functional/states'
import { State } from '../functional/types'

/**
 * Process one action
 * @param state
 * @param action
 */
function reduceOne (state: State, action: types.BaseAction): State {
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
    case types.UPDATE_STATE:
      return common.updateState(state, action as types.UpdateStateAction)
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
    case types.SUBMIT:
      return common.submit(state, action as types.SubmitAction)
    case types.START_LINK_TRACK:
      return common.startLinkTrack(state)
    case types.UPDATE_SESSION_STATUS:
      return common.updateSessionStatus(
        state, action as types.UpdateSessionStatusAction)
    case types.NULL:
      return state
    default:
  }
  return state
}

/**
 * Reducer
 * @param {State} currentState
 * @param {AnyAction} action
 * @return {State}
 */
export const reducer: Reducer<State> = (
    currentState: State = makeState(),
    action: AnyAction): State => {
  let state = currentState
  if (action.type === types.SEQUENTIAL) {
    (action as types.SequentialAction).actions.forEach((element) => {
      state = reduceOne(state, element)
    })
  } else {
    state = reduceOne(state, action as types.BaseAction)
  }
  return state
}

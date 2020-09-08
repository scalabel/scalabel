import { AnyAction, Reducer } from "redux"

import * as actionConsts from "../const/action"
import * as common from "../functional/common"
import { makeState } from "../functional/states"
import * as actionTypes from "../types/action"
import { State } from "../types/state"

/**
 * Process one action
 *
 * @param state
 * @param action
 */
function reduceOne(state: State, action: actionTypes.BaseAction): State {
  switch (action.type) {
    case actionConsts.INIT_SESSION:
      return common.initSession(state)
    case actionConsts.CHANGE_SELECT:
      return common.changeSelect(
        state,
        action as actionTypes.ChangeSelectAction
      )
    case actionConsts.LOAD_ITEM:
      return common.loadItem(state, action as actionTypes.LoadItemAction)
    case actionConsts.UPDATE_ALL:
      return common.updateAll(state)
    case actionConsts.UPDATE_TASK:
      return common.updateTask(state, action as actionTypes.UpdateTaskAction)
    case actionConsts.UPDATE_STATE:
      return common.updateState(state, action as actionTypes.UpdateStateAction)
    case actionConsts.ADD_LABELS:
      return common.addLabels(state, action as actionTypes.AddLabelsAction)
    case actionConsts.ADD_TRACK:
      return common.addTrack(state, action as actionTypes.AddTrackAction)
    case actionConsts.CHANGE_SHAPES:
      return common.changeShapes(
        state,
        action as actionTypes.ChangeShapesAction
      )
    case actionConsts.CHANGE_LABELS:
      return common.changeLabels(
        state,
        action as actionTypes.ChangeLabelsAction
      )
    case actionConsts.LINK_LABELS:
      return common.linkLabels(state, action as actionTypes.LinkLabelsAction)
    case actionConsts.UNLINK_LABELS:
      return common.unlinkLabels(
        state,
        action as actionTypes.UnlinkLabelsAction
      )
    case actionConsts.MERGE_TRACKS:
      return common.mergeTracks(state, action as actionTypes.MergeTrackAction)
    case actionConsts.DELETE_LABELS:
      return common.deleteLabels(
        state,
        action as actionTypes.DeleteLabelsAction
      )
    case actionConsts.ADD_VIEWER_CONFIG:
      return common.addViewerConfig(
        state,
        action as actionTypes.AddViewerConfigAction
      )
    case actionConsts.UPDATE_PANE:
      return common.updatePane(state, action as actionTypes.UpdatePaneAction)
    case actionConsts.SPLIT_PANE:
      return common.splitPane(state, action as actionTypes.SplitPaneAction)
    case actionConsts.DELETE_PANE:
      return common.deletePane(state, action as actionTypes.DeletePaneAction)
    case actionConsts.CHANGE_VIEWER_CONFIG:
      return common.changeViewerConfig(
        state,
        action as actionTypes.ChangeViewerConfigAction
      )
    case actionConsts.SUBMIT:
      return common.submit(state, action as actionTypes.SubmitAction)
    case actionConsts.START_LINK_TRACK:
      return common.startLinkTrack(state)
    case actionConsts.UPDATE_SESSION_STATUS:
      return common.updateSessionStatus(
        state,
        action as actionTypes.UpdateSessionStatusAction
      )
    case actionConsts.NULL:
      return state
    default:
  }
  return state
}

/**
 * Reducer
 *
 * @param {State} currentState
 * @param {AnyAction} action
 * @returns {State}
 */
export const reducer: Reducer<State> = (
  currentState: State | undefined,
  action: AnyAction
): State => {
  let state = currentState !== undefined ? currentState : makeState()
  if (action.type === actionConsts.SEQUENTIAL) {
    ;(action as actionTypes.SequentialAction).actions.forEach((element) => {
      state = reduceOne(state, element)
    })
  } else {
    state = reduceOne(state, action as actionTypes.BaseAction)
  }
  return state
}

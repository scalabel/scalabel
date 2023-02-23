import { AnyAction, Reducer } from "redux"

import * as actionConsts from "../const/action"
import * as common from "../functional/common"
import * as span3d from "../functional/span3d"
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
    case actionConsts.SPLIT_TRACK:
      return common.splitTrack(state, action as actionTypes.SplitTrackAction)
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
    case actionConsts.STOP_LINK_TRACK:
      return common.stopLinkTrack(state)
    case actionConsts.UPDATE_POLYGON2D_BOUNDARY_CLONE:
      return common.updatePolygon2DBoundaryCloneStatus(
        state,
        action as actionTypes.UpdatePolygon2DBoundaryCloneStatusAction
      )
    case actionConsts.UPDATE_SESSION_STATUS:
      return common.updateSessionStatus(
        state,
        action as actionTypes.UpdateSessionStatusAction
      )
    case actionConsts.CHANGE_SESSION_MODE:
      return common.changeSessionMode(
        state,
        action as actionTypes.ChangeSessionModeAction
      )
    //todo for overlay
    case actionConsts.CHANGE_OVERLAYS:
      return common.changeOverlays(
        state,
        action as actionTypes.ChangeOverlaysAction
      )
    case actionConsts.CHANGE_OVERLAY_TRANSPARENCY:
      return common.changeOverlayTransparency(
        state,
        action as actionTypes.ChangeOverlayTransparencyAction
      )
    case actionConsts.CHANGE_RADAR_STATUS:
      return common.changeRadarStatus(
        state,
        action as actionTypes.ChangeRadarStatusAction
      )
    case actionConsts.ADD_ALERT:
      return common.addAlert(state, action as actionTypes.AddAlertAction)
    case actionConsts.CLOSE_ALERT:
      return common.removeAlert(state, action as actionTypes.RemoveAlertAction)
    case actionConsts.ACTIVATE_SPAN:
      return span3d.activateSpan(state)
    case actionConsts.DEACTIVATE_SPAN:
      return span3d.deactivateSpan(state)
    case actionConsts.REGISTER_SPAN_POINT:
      return span3d.registerSpanPoint(state)
    case actionConsts.RESET_SPAN:
      return span3d.resetSpan(state)
    case actionConsts.PAUSE_SPAN:
      return span3d.pauseSpan(state)
    case actionConsts.RESUME_SPAN:
      return span3d.resumeSpan(state)
    case actionConsts.UNDO_SPAN:
      return span3d.undoSpan(state)
    case actionConsts.TOGGLE_GROUND_PLANE:
      return common.toggleGroundPlane(state)
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

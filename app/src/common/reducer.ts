import { AnyAction, Reducer } from "redux"

import * as actionConsts from "../const/action"
import * as common from "../functional/common"
import * as span3d from "../functional/span3d"
import { makeState } from "../functional/states"
import * as actionTypes from "../types/action"
import { State } from "../types/state"
import { Severity } from "../types/common"
import { uid } from "../common/uid"

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

const readonlyActions = new Set<string>([
  actionConsts.INIT_SESSION,
  actionConsts.CHANGE_SELECT,
  actionConsts.LOAD_ITEM,
  actionConsts.UPDATE_ALL,
  actionConsts.UPDATE_STATE,
  actionConsts.ADD_VIEWER_CONFIG,
  actionConsts.UPDATE_PANE,
  actionConsts.SPLIT_PANE,
  actionConsts.DELETE_PANE,
  actionConsts.CHANGE_VIEWER_CONFIG,
  actionConsts.UPDATE_SESSION_STATUS,
  actionConsts.CHANGE_SESSION_MODE,
  actionConsts.ADD_ALERT,
  actionConsts.CLOSE_ALERT,
  actionConsts.ACTIVATE_SPAN,
  actionConsts.DEACTIVATE_SPAN,
  actionConsts.REGISTER_SPAN_POINT,
  actionConsts.RESET_SPAN,
  actionConsts.PAUSE_SPAN,
  actionConsts.RESUME_SPAN,
  actionConsts.UNDO_SPAN,
  actionConsts.TOGGLE_GROUND_PLANE,
  actionConsts.NULL
])

export const readonlyReducer: Reducer<State> = (
  currentState: State | undefined,
  action: AnyAction
): State => {
  const atype = action.type as string
  // Redux itself wills send some actions. See
  // https://github.com/reduxjs/redux/blob/v4.1.0/src/utils/actionTypes.js
  // Do note that upgrading Redux may cause the following check to fail.
  if (atype.startsWith("@@redux")) {
    return reducer(currentState, action)
  }

  const state = currentState ?? makeState()

  // Remark(hxu): Using state to handle alert does not make sense to me.
  // Nevertheless, we use the existing framework to achieve the purpose.
  const readonlyAlert = addReadonlyAlertAction(state)

  // Don't show an alert, but leave the code for potential future reference.
  const canAlert = isFrontend() && false

  let finalAction: AnyAction | undefined
  if (atype === actionConsts.SEQUENTIAL) {
    const seq = action as actionTypes.SequentialAction
    const filtered = seq.actions.filter((a) => readonlyActions.has(a.type))
    const n = seq.actions.length
    const invalid = n !== filtered.length
    if (invalid) {
      const as = seq.actions.filter((a) => !readonlyActions.has(a.type))
      const types = as.map((a) => a.type).join(",")
      console.warn(`attempt to apply action ${types} in readonly mode`)
    }
    const alert = canAlert && n > 0
    const actions = invalid && alert ? [...filtered, readonlyAlert] : filtered
    seq.actions = actions
    finalAction = actions.length > 0 ? seq : undefined
  } else {
    const allow = readonlyActions.has(atype)
    if (allow) {
      finalAction = action
    } else {
      console.warn(`attempt to apply action ${atype} in readonly mode`)
      finalAction = canAlert ? readonlyAlert : undefined
    }
  }

  return finalAction != null ? reducer(currentState, finalAction) : state
}

/**
 * Create an AddAlertAction to indicate being in readonly mode.
 *
 * @param state
 */
function addReadonlyAlertAction(state: State): actionTypes.AddAlertAction {
  return {
    actionId: uid(),
    sessionId: state.session.id,
    userId: state.user.id,
    type: actionConsts.ADD_ALERT,
    timestamp: Date.now(),
    alert: {
      id: uid(),
      severity: Severity.WARNING,
      message: "Can not perform this action in readonly mode.",
      timeout: 1000
    }
  }
}

/**
 * Check if the script is running in frontend.
 *
 */
function isFrontend(): boolean {
  return typeof window !== "undefined"
}

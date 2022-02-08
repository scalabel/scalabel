import * as actionConsts from "../const/action"
import * as actionTypes from "../types/action"
import { makeBaseAction } from "./common"

/** Activate box spanning mode */
export function activateSpan(): actionTypes.ActivateSpanAction {
  return makeBaseAction(actionConsts.ACTIVATE_SPAN)
}

/** Register span box as new 3D label */
export function deactivateSpan(): actionTypes.DeactivateSpanAction {
  return makeBaseAction(actionConsts.DEACTIVATE_SPAN)
}

/**
 * Register new point in span box
 *
 * @param mX - mouse X
 * @param mY - mouse Y
 */
export function registerSpanPoint(): actionTypes.RegisterSpanPointAction {
  return makeBaseAction(actionConsts.REGISTER_SPAN_POINT)
}

/** Register updated span point */
export function updateSpanPoint(): actionTypes.UpdateSpanPoint {
  return makeBaseAction(actionConsts.UPDATE_SPAN_POINT)
}

/** Reset span box generation */
export function resetSpan(): actionTypes.ResetSpanAction {
  return makeBaseAction(actionConsts.RESET_SPAN)
}

/** Reset span box generation */
export function pauseSpan(): actionTypes.PauseSpanAction {
  return makeBaseAction(actionConsts.PAUSE_SPAN)
}

/** Reset span box generation */
export function resumeSpan(): actionTypes.ResumeSpanAction {
  return makeBaseAction(actionConsts.RESUME_SPAN)
}

/** Undo span point registration */
export function undoSpan(): actionTypes.UndoSpanAction {
  return makeBaseAction(actionConsts.UNDO_SPAN)
}

/** Show/hide ground plane */
export function toggleGroundPlane(): actionTypes.ToggleGroundPlaneAction {
  return makeBaseAction(actionConsts.TOGGLE_GROUND_PLANE)
}

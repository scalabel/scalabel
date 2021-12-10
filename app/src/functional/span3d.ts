import * as actionTypes from "../types/action"
import { State } from "../types/state"
import { updateObject } from "./util"
import { Span3D } from "../drawable/3d/box_span/span3d"

/**
 * Activate box spanning mode
 *
 * @param state
 */
export function activateSpan(state: State): State {
  const oldSession = state.session
  const newSession = updateObject(oldSession, {
    ...state.session,
    isBoxSpan: true,
    boxSpan: new Span3D()
  })
  return updateObject(state, {
    session: newSession
  })
}

/**
 * Deactivate box spanning mode
 *
 * @param state
 */
export function deactivateSpan(state: State): State {
  const oldSession = state.session
  const newSession = updateObject(oldSession, {
    ...state.session,
    isBoxSpan: false,
    boxSpan: null
  })
  return updateObject(state, {
    session: newSession
  })
}

/**
 * Update temporary point in span box
 *
 * @param state
 * @param action
 */
export function updateSpanPoint(
  state: State,
  action: actionTypes.UpdateSpanPointAction
): State {
  const oldSession = state.session
  const newBox = oldSession.boxSpan
  if (newBox !== null) {
    newBox.updatePointTmp(action.point, action.mousePos)
  }
  const newSession = updateObject(oldSession, {
    ...state.session,
    boxSpan: newBox
  })
  return updateObject(state, {
    session: newSession
  })
}

/**
 * Register new point in span box
 *
 * @param state
 */
export function registerSpanPoint(state: State): State {
  const oldSession = state.session
  const newBox = oldSession.boxSpan
  if (newBox !== null) {
    newBox.registerPoint()
  }
  const newSession = updateObject(oldSession, {
    ...state.session,
    boxSpan: newBox
  })
  return updateObject(state, {
    session: newSession
  })
}

/**
 * Reset span box
 *
 * @param state
 */
export function resetSpan(state: State): State {
  const oldSession = state.session
  const newSession = updateObject(oldSession, {
    ...state.session,
    boxSpan: new Span3D()
  })
  return updateObject(state, {
    session: newSession
  })
}

/**
 * Pause box spanning mode
 *
 * @param state
 */
export function pauseSpan(state: State): State {
  const oldSession = state.session
  const newSession = updateObject(oldSession, {
    ...state.session,
    isBoxSpan: false
  })
  return updateObject(state, {
    session: newSession
  })
}

/**
 * Resume box spanning mode
 *
 * @param state
 */
export function resumeSpan(state: State): State {
  const oldSession = state.session
  const newSession = updateObject(oldSession, {
    ...state.session,
    isBoxSpan: true
  })
  return updateObject(state, {
    session: newSession
  })
}

/**
 * Undo span point registration
 *
 * @param state
 */
export function undoSpan(state: State): State {
  const oldSession = state.session
  const box = oldSession.boxSpan
  if (box !== null) {
    box.removeLastPoint()
  }
  const newSession = updateObject(oldSession, {
    ...state.session,
    boxSpan: box
  })
  return updateObject(state, {
    session: newSession
  })
}

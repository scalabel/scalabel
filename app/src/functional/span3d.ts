// import * as actionTypes from "../types/action"
import { State } from "../types/state"
import { updateObject } from "./util"
import { Span3D } from "../drawable/3d/box_span/span3d"

/**
 * Activate box spanning mode
 *
 * @param state
 */
export function activateSpan(state: State): State {
  const oldInfo3D = state.session.info3D
  const newInfo3D = updateObject(oldInfo3D, {
    ...oldInfo3D,
    isBoxSpan: true,
    boxSpan: new Span3D()
  })
  const oldSession = state.session
  const newSession = updateObject(oldSession, {
    ...state.session,
    info3D: newInfo3D
  })
  return updateObject(state, {
    ...state,
    session: newSession
  })
}

/**
 * Deactivate box spanning mode
 *
 * @param state
 */
export function deactivateSpan(state: State): State {
  const oldInfo3D = state.session.info3D
  const newInfo3D = updateObject(oldInfo3D, {
    ...oldInfo3D,
    isBoxSpan: false,
    boxSpan: null
  })
  const oldSession = state.session
  const newSession = updateObject(oldSession, {
    ...state.session,
    info3D: newInfo3D
  })
  return updateObject(state, {
    ...state,
    session: newSession
  })
}

/**
 * Register new point in span box
 *
 * @param state
 */
export function registerSpanPoint(state: State): State {
  const oldInfo3D = state.session.info3D
  const newBox = oldInfo3D.boxSpan
  if (newBox !== null) {
    newBox.registerPoint()
  }
  const newInfo3D = updateObject(oldInfo3D, {
    ...oldInfo3D,
    boxSpan: newBox
  })
  const oldSession = state.session
  const newSession = updateObject(oldSession, {
    ...state.session,
    info3D: newInfo3D
  })
  return updateObject(state, {
    ...state,
    session: newSession
  })
}

/**
 * Reset span box
 *
 * @param state
 */
export function resetSpan(state: State): State {
  const oldInfo3D = state.session.info3D
  const newInfo3D = updateObject(oldInfo3D, {
    ...state.session.info3D,
    boxSpan: new Span3D()
  })
  const oldSession = state.session
  const newSession = updateObject(oldSession, {
    ...state.session,
    info3D: newInfo3D
  })
  return updateObject(state, {
    ...state,
    session: newSession
  })
}

/**
 * Pause box spanning mode
 *
 * @param state
 */
export function pauseSpan(state: State): State {
  const oldInfo3D = state.session.info3D
  const newInfo3D = updateObject(oldInfo3D, {
    ...state.session.info3D,
    isBoxSpan: false
  })
  const oldSession = state.session
  const newSession = updateObject(oldSession, {
    ...state.session,
    info3D: newInfo3D
  })
  return updateObject(state, {
    ...state,
    session: newSession
  })
}

/**
 * Resume box spanning mode
 *
 * @param state
 */
export function resumeSpan(state: State): State {
  const oldInfo3D = state.session.info3D
  const newInfo3D = updateObject(oldInfo3D, {
    ...state.session.info3D,
    isBoxSpan: true
  })
  const oldSession = state.session
  const newSession = updateObject(oldSession, {
    ...state.session,
    info3D: newInfo3D
  })
  return updateObject(state, {
    ...state,
    session: newSession
  })
}

/**
 * Undo span point registration
 *
 * @param state
 */
export function undoSpan(state: State): State {
  const oldInfo3D = state.session.info3D
  const newBox = oldInfo3D.boxSpan
  if (newBox !== null) {
    newBox.removeLastPoint()
  }
  const newInfo3D = updateObject(oldInfo3D, {
    ...state.session.info3D,
    boxSpan: newBox
  })
  const oldSession = state.session
  const newSession = updateObject(oldSession, {
    ...state.session,
    info3D: newInfo3D
  })
  return updateObject(state, {
    ...state,
    session: newSession
  })
}

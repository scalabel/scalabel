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
    boxSpan: true
  })
  const oldTask = state.task
  const newTask = updateObject(oldTask, {
    ...state.task,
    boxSpan: new Span3D()
  })
  return updateObject(state, {
    session: newSession,
    task: newTask
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
    boxSpan: false
  })
  const oldTask = state.task
  const newTask = updateObject(oldTask, {
    ...state.task,
    boxSpan: null
  })
  return updateObject(state, {
    session: newSession,
    task: newTask
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
  const oldTask = state.task
  const newBox = oldTask.boxSpan
  if (newBox !== null) {
    newBox.updatePointTmp(action.point, action.mousePos)
  }
  const newTask = updateObject(oldTask, {
    ...state.task,
    boxSpan: newBox
  })
  return updateObject(state, {
    task: newTask
  })
}

/**
 * Register new point in span box
 *
 * @param state
 */
export function registerSpanPoint(state: State): State {
  const oldTask = state.task
  const newBox = oldTask.boxSpan
  if (newBox !== null) {
    newBox.registerPoint()
  }
  const newTask = updateObject(oldTask, {
    ...state.task,
    boxSpan: newBox
  })
  return updateObject(state, {
    task: newTask
  })
}

/**
 * Reset span box
 *
 * @param state
 */
export function resetSpan(state: State): State {
  const oldTask = state.task
  const newTask = updateObject(oldTask, {
    ...state.task,
    boxSpan: new Span3D()
  })
  return updateObject(state, {
    task: newTask
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
    boxSpan: false
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
    boxSpan: true
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
  const oldTask = state.task
  const box = oldTask.boxSpan
  if (box !== null) {
    box.removeLastPoint()
  }
  const newTask = updateObject(oldTask, {
    ...state.task,
    boxSpan: box
  })
  return updateObject(state, {
    task: newTask
  })
}

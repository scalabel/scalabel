import { BaseAction as actionConsts } from "../types/action"

export const INIT_SESSION = "INIT_SESSION"
export const CHANGE_SELECT = "CHANGE_SELECT"
export const CHANGE_SESSION_MODE = "CHANGE_SESSION_MODE"
export const CHANGE_OVERLAYS = "CHANGE_OVERLAYS"
export const CHANGE_OVERLAY_TRANSPARENCY = "CHANGE_OVERLAY_TRANSPARENCY"
export const CHANGE_RADAR_STATUS = "CHANGE_RADAR_STATUS"
export const LOAD_ITEM = "LOAD_ITEM"
export const UPDATE_ALL = "UPDATE_ALL"
export const UPDATE_TASK = "UPDATE_TASK"
export const UPDATE_STATE = "UPDATE_STATE"
export const SUBMIT = "SUBMIT"
export const UPDATE_SESSION_STATUS = "UPDATE_SESSION_STATUS"
export const ADD_ALERT = "ADD_ALERT"
export const CLOSE_ALERT = "CLOSE_ALERT"
export const NULL = "NULL"

// Item Level
export const ADD_LABELS = "ADD_LABELS"
export const CHANGE_SHAPES = "CHANGE_SHAPES"
export const CHANGE_LABELS = "CHANGE_LABELS"
export const LINK_LABELS = "LINK_LABELS"
export const UNLINK_LABELS = "UNLINK_LABELS"
export const DELETE_LABELS = "DELETE_LABELS"

export const ADD_TRACK = "ADD_TRACK"
export const MERGE_TRACKS = "MERGE_TRACKS"
export const SPLIT_TRACK = "SPLIT_TRACK"

export const ACTIVATE_SPAN = "ACTIVATE_SPAN"
export const DEACTIVATE_SPAN = "DEACTIVATE_SPAN"
export const REGISTER_SPAN_POINT = "REGISTER_SPAN_POINT"
export const UPDATE_SPAN_POINT = "UPDATE_SPAN_POINT"
export const RESET_SPAN = "RESET_SPAN"
export const PAUSE_SPAN = "PAUSE_SPAN"
export const RESUME_SPAN = "RESUME_SPAN"
export const UNDO_SPAN = "UNDO_SPAN"

export const SET_GROUND_PLANE = "SET_GROUND_PLANE"
export const TOGGLE_GROUND_PLANE = "TOGGLE_GROUND_PLANE"

// View Level
export const ADD_VIEWER_CONFIG = "ADD_VIEWER_CONFIG"
export const CHANGE_VIEWER_CONFIG = "CHANGE_VIEWER_CONFIG"
export const UPDATE_PANE = "UPDATE_PANE"
export const SPLIT_PANE = "SPLIT_PANE"
export const DELETE_PANE = "DELETE_PANE"
export const START_LINK_TRACK = "START_LINK_TRACK"
export const STOP_LINK_TRACK = "STOP_LINK_TRACK"
export const UPDATE_POLYGON2D_BOUNDARY_CLONE = "UPDATE_POLYGON2D_BOUNDARY_CLONE"

// Sync based events
export const REGISTER_SESSION = "REGISTER_SESSION"
export const RECEIVE_BROADCAST = "RECEIVE_BROADCAST"
export const CONNECT = "CONNECT"
export const DISCONNECT = "DISCONNECT"
export const SAVE = "SAVE"

// A sequence of actions
export const SEQUENTIAL = "SEQUENTIAL"
/**
 * These are actions that should be shared between sessions/users
 * UPDATE_TASK deliberately not included because its used for local updates
 */
const TASK_ACTION_TYPES = [
  ADD_LABELS,
  ADD_TRACK,
  CHANGE_SHAPES,
  CHANGE_LABELS,
  LINK_LABELS,
  DELETE_LABELS,
  MERGE_TRACKS,
  SPLIT_TRACK,
  SUBMIT
]

/**
 * Checks if the action modifies task
 *
 * @param action
 */
export function isTaskAction(action: actionConsts): boolean {
  return TASK_ACTION_TYPES.includes(action.type)
}

/**
 * These actions are intercepted by sync middleware, not used to update state
 * They trigger an interaction with the backend
 */
const SYNC_ACTION_TYPES = [
  REGISTER_SESSION,
  RECEIVE_BROADCAST,
  CONNECT,
  DISCONNECT,
  SAVE
]

/**
 * Checks if the action should be intercepted by the sync middleware
 *
 * @param action
 */
export function isSyncAction(action: actionConsts): boolean {
  return SYNC_ACTION_TYPES.includes(action.type)
}

/**
 * Checks if the action list contains a submit action
 *
 * @param actions
 */
export function hasSubmitAction(actions: actionConsts[]): boolean {
  for (const action of actions) {
    if (action.type === SUBMIT) {
      return true
    }
  }
  return false
}

/**
 * These actions are used to update state for 3D functions
 */
const INFO3D_ACTION_TYPES = [
  ACTIVATE_SPAN,
  DEACTIVATE_SPAN,
  PAUSE_SPAN,
  REGISTER_SPAN_POINT,
  UPDATE_SPAN_POINT,
  RESET_SPAN,
  RESUME_SPAN,
  SET_GROUND_PLANE,
  TOGGLE_GROUND_PLANE,
  UNDO_SPAN
]

/**
 * Checks if the action modifies 3D info
 *
 * @param action
 */
export function isInfo3DAction(action: actionConsts): boolean {
  return INFO3D_ACTION_TYPES.includes(action.type)
}

/**
 * These actions should not be broadcast outside the local session
 */
const SESSION_ACTION_TYPES = [
  UPDATE_SESSION_STATUS,
  CHANGE_SESSION_MODE,
  CHANGE_SELECT
]

/**
 * Checks if the action modifies session
 *
 * @param action
 */
export function isSessionAction(action: actionConsts): boolean {
  return SESSION_ACTION_TYPES.includes(action.type)
}

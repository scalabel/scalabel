import { BaseAction as actionConsts } from "../types/action"

export const INIT_SESSION = "INIT_SESSION"
export const CHANGE_SELECT = "CHANGE_SELECT"
export const LOAD_ITEM = "LOAD_ITEM"
export const UPDATE_ALL = "UPDATE_ALL"
export const UPDATE_TASK = "UPDATE_TASK"
export const UPDATE_STATE = "UPDATE_STATE"
export const SUBMIT = "SUBMIT"
export const UPDATE_SESSION_STATUS = "UPDATE_SESSION_STATUS"
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

// View Level
export const ADD_VIEWER_CONFIG = "ADD_VIEWER_CONFIG"
export const CHANGE_VIEWER_CONFIG = "CHANGE_VIEWER_CONFIG"
export const UPDATE_PANE = "UPDATE_PANE"
export const SPLIT_PANE = "SPLIT_PANE"
export const DELETE_PANE = "DELETE_PANE"
export const START_LINK_TRACK = "START_LINK_TRACK"

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
  CHANGE_SHAPES,
  CHANGE_LABELS,
  LINK_LABELS,
  DELETE_LABELS,
  ADD_TRACK,
  MERGE_TRACKS,
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
 * These actions should not be broadcast outside the local session
 */
const SESSION_ACTION_TYPES = [UPDATE_SESSION_STATUS, CHANGE_SELECT]

/**
 * Checks if the action modifies session
 *
 * @param action
 */
export function isSessionAction(action: actionConsts): boolean {
  return SESSION_ACTION_TYPES.includes(action.type)
}

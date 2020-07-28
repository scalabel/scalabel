import _ from 'lodash'
import * as THREE from 'three'
import { Window } from '../components/window'
import { Label2DList } from '../drawable/2d/label2d_list'
import { Label3DList } from '../drawable/3d/label3d_list'
import * as actionTypes from '../types/action'
import { FullStore, ThunkActionType } from '../types/common'
import { State } from '../types/functional'
import { configureStore } from './configure_store'
import { GetStateFunc, SimpleStore } from './simple_store'
import { Track } from './track'

/**
 * Singleton session class
 */
class Session {
  /** The store to save states */
  public store: FullStore
  /** Images of the session */
  public images: Array<{[id: number]: HTMLImageElement}>
  /** Point cloud */
  public pointClouds: Array<{[id: number]: THREE.BufferGeometry}>
  /** 2d label list */
  public label2dList: Label2DList
  /** 3d label list */
  public label3dList: Label3DList
  /** map between track id and track objects */
  public tracks: {[trackId: string]: Track}
  /** id of the viewer that the mouse is currently hovering over */
  public activeViewerId: number
  /** The window component */
  public window?: Window
  /** if in test mode, needed for integration and end to end testing */
  // TODO: when we move to node move this into state
  public testMode: boolean

  constructor () {
    this.images = []
    this.pointClouds = []
    this.label2dList = new Label2DList()
    this.label3dList = new Label3DList()
    this.tracks = {}
    this.activeViewerId = -1
    this.testMode = false
    this.store = configureStore({})
  }

  /**
   * Get current state in store
   * @return {State}
   */
  public getState (): State {
    return this.store.getState().present
  }

  /**
   * Get a simple store instance type for use without Session
   */
  public getSimpleStore (): SimpleStore {
    return new SimpleStore(this.getState.bind(this), this.dispatch.bind(this))
  }

  /**
   * Get the id of the current session
   */
  public get id (): string {
    return this.getState().session.id
  }

  /**
   * Get the number of items in the current session
   */
  public get numItems (): number {
    return Math.max(this.images.length, this.pointClouds.length)
  }

  /**
   * Wrapper for redux store dispatch of actions
   * @param {actionTypes.ActionType} action: action description
   */
  public dispatch (action: actionTypes.ActionType | ThunkActionType) {
    if (action.hasOwnProperty('type')) {
      this.store.dispatch(action as actionTypes.ActionType)
    } else {
      this.store.dispatch(action as ThunkActionType)
    }
  }

  /**
   * Subscribe all the controllers to the states
   * @param {Function} callback: view component
   */
  public subscribe (callback: () => void) {
    this.store.subscribe(callback)
  }
}

const session = new Session()

/**
 * extract state getter from the session
 */
export function getStateGetter (): GetStateFunc {
  return session.getSimpleStore().getter()
}

/**
 * Get state from the global session instance
 */
export function getState (): State {
  return session.getState()
}

/**
 * Dispatch the action to the global session instance
 */
export function dispatch (action: actionTypes.ActionType | ThunkActionType):
  void {
  return session.dispatch(action)
}

/**
 * Get the simple store object
 */
export function getStore (): SimpleStore {
  return session.getSimpleStore()
}

export default session

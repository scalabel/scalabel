import _ from 'lodash'
import { Store } from 'redux'
import { StateWithHistory } from 'redux-undo'
import * as THREE from 'three'
import * as types from '../action/types'
import { Window } from '../components/window'
import { Label2DList } from '../drawable/2d/label2d_list'
import { Label3DList } from '../drawable/3d/label3d_list'
import { State } from '../functional/types'
import { configureStore } from './configure_store'
import { Track } from './track'

export const enum ConnectionStatus {
  NOTIFY_SAVED, SAVED, SAVING, RECONNECTING, UNSAVED
}

/**
 * Singleton session class
 */
class Session {
  /** The store to save states */
  public store: Store<StateWithHistory<State>>
  /** Images of the session */
  public images: Array<{[id: number]: HTMLImageElement}>
  /** Point cloud */
  public pointClouds: Array<{[id: number]: THREE.BufferGeometry}>
  /** 2d label list */
  public label2dList: Label2DList
  /** 3d label list */
  public label3dList: Label3DList
  /** map between track id and track objects */
  public tracks: {[trackId: number]: Track}
  /** whether tracking is enabled */
  public tracking: boolean
  /** whether track linking is enabled */
  public trackLinking: boolean
  /** id of the viewer that the mouse is currently hovering over */
  public activeViewerId: number
  /** The window component */
  public window?: Window
  /** Whether autosave is enabled */
  public autosave: boolean
  /** Dev mode */
  public devMode: boolean
  /** if in test mode, needed for integration and end to end testing */
  // TODO: when we move to node move this into state
  public testMode: boolean
  /** Connection status for saving */
  public status: ConnectionStatus
  /** Previous connection status */
  public prevStatus: ConnectionStatus
  /** Number of times connection status has changed */
  public statusChangeCount: number
  /** Overwriteable function that adds side effects to state change */
  public applyStatusEffects: () => void

  constructor () {
    this.images = []
    this.pointClouds = []
    this.label2dList = new Label2DList()
    this.label3dList = new Label3DList()
    this.tracks = {}
    this.tracking = true
    this.trackLinking = false
    this.activeViewerId = -1
    this.status = ConnectionStatus.UNSAVED
    this.prevStatus = ConnectionStatus.UNSAVED
    this.statusChangeCount = 0
    this.autosave = false
    // TODO: make it configurable in the url
    this.devMode = true
    this.applyStatusEffects = () => { return }
    this.testMode = false
    this.store = configureStore({}, this.devMode)
  }

  /**
   * Get current state in store
   * @return {State}
   */
  public getState (): State {
    return this.store.getState().present
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
   * Wrapper for redux store dispatch
   * @param {types.ActionType} action: action description
   */
  public dispatch (action: types.ActionType): void {
    this.store.dispatch(action)
  }

  /**
   * Subscribe all the controllers to the states
   * @param {Function} callback: view component
   */
  public subscribe (callback: () => void) {
    this.store.subscribe(callback)
  }

  /**
   * Update the status, then call overwritable function
   * This should update any parts of the view that depend on status
   * @param {ConnectionStatus} newStatus: new value of status
   */
  public updateStatus (newStatus: ConnectionStatus): ConnectionStatus {
    this.prevStatus = this.status
    this.status = newStatus
    // update mod 1000 since only nearby differences are important, not total
    this.statusChangeCount = (this.statusChangeCount + 1) % 1000
    this.applyStatusEffects()
    return newStatus
  }
}

export default new Session()

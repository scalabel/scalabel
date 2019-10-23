import _ from 'lodash'
import { Store } from 'redux'
import { StateWithHistory } from 'redux-undo'
import * as THREE from 'three'
import * as types from '../action/types'
import { Window } from '../components/window'
import { State } from '../functional/types'
import { configureStore } from './configure_store'
import { Track } from './track'

export const enum ConnectionStatus {
  SAVED, SAVING, RECONNECTING, UNSAVED
}

/**
 * Singleton session class
 */
class Session {
  /** The store to save states */
  public store: Store<StateWithHistory<State>>
  /** Images of the session */
  public images: HTMLImageElement[]
  /** Point cloud */
  public pointClouds: THREE.Points[]
  /** map between track id and track objects */
  public tracks: {[trackId: number]: Track}
  /** Item type: image, point cloud */
  public itemType: string
  /** whether tracking is enabled */
  public tracking: boolean
  /** Current tracking policy type */
  public currentPolicyType: string
  /** The window component */
  public window?: Window
  /** Dev mode */
  public devMode: boolean
  /** if in test mode, needed for integration and end to end testing */
  // TODO: when we move to node move this into state
  public testMode: boolean
  /** Connection status for saving */
  public status: ConnectionStatus
  /** Overwriteable function that adds side effects to state change */
  public applyStatusEffects: () => void

  constructor () {
    this.images = []
    this.pointClouds = []
    this.tracks = {}
    this.itemType = ''
    this.tracking = true
    this.currentPolicyType = ''
    this.status = ConnectionStatus.UNSAVED
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
    return (this.itemType === 'image') ? this.images.length :
      this.pointClouds.length
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
  public updateStatusDisplay (newStatus: ConnectionStatus): ConnectionStatus {
    this.status = newStatus
    this.applyStatusEffects()
    return newStatus
  }
}

export default new Session()

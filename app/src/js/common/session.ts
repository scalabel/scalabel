import _ from 'lodash'
import { Store } from 'redux'
import { StateWithHistory } from 'redux-undo'
import * as THREE from 'three'
import * as types from '../action/types'
import { Window } from '../components/window'
import { State } from '../functional/types'
import { configureStore } from './configure_store'

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
  /** Item type: image, point cloud */
  public itemType: string
  /** The window component */
  public window?: Window
  /** Dev mode */
  public devMode: boolean
  /** Connection status */
  public status: ConnectionStatus

  /**
   * no-op for state initialization
   */
  constructor () {
    this.images = []
    this.pointClouds = []
    this.itemType = ''
    // TODO: make it configurable in the url
    this.devMode = true
    this.store = configureStore({}, this.devMode)
    this.status = ConnectionStatus.UNSAVED
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
    return this.getState().config.sessionId
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
}

export default new Session()

import _ from 'lodash'
import { AnyAction, Store } from 'redux'
import { ThunkAction, ThunkDispatch } from 'redux-thunk'
import { StateWithHistory } from 'redux-undo'
import * as THREE from 'three'
import * as types from '../action/types'
import { Window } from '../components/window'
import { Label2DList } from '../drawable/2d/label2d_list'
import { Label3DList } from '../drawable/3d/label3d_list'
import { State } from '../functional/types'
import { configureStore } from './configure_store'
import { Track } from './track/track'

/**
 * Singleton session class
 */
class Session {
  /** The store to save states */
  public store: Store<StateWithHistory<State>, AnyAction> & {
    /** Thunk dispatch used for redux-thunk async actions */
    dispatch: ThunkDispatch<StateWithHistory<State>, undefined, AnyAction>;
  }
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
  /** Whether bots are enabled */
  public bots: boolean

  constructor () {
    this.images = []
    this.pointClouds = []
    this.label2dList = new Label2DList()
    this.label3dList = new Label3DList()
    this.tracks = {}
    this.tracking = true
    this.trackLinking = false
    this.activeViewerId = -1
    this.autosave = false
    // TODO: make it configurable in the url
    this.devMode = false
    this.testMode = false
    this.store = configureStore({}, this.devMode)
    this.bots = false
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
   * Wrapper for redux store dispatch of actions
   * @param {types.ActionType} action: action description
   */
  public dispatch (action: types.ActionType | ThunkAction<
    void, StateWithHistory<State>, void, types.ActionType>) {
    if (action.hasOwnProperty('type')) {
      this.store.dispatch(action as types.ActionType)
    } else {
      this.store.dispatch(action as ThunkAction<
        void, StateWithHistory<State>, void, types.ActionType>)
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

export default new Session()

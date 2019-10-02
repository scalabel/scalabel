import _ from 'lodash'
import { Dispatch, Middleware } from 'redux'
import io from 'socket.io-client'
import { updateTask } from '../action/common'
import * as types from '../action/types'
import { State } from '../functional/types'
import { EventName } from '../server/hub'
import Session, { ConnectionStatus } from './session'

/**
 * Synchronizes data with other sessions
 */
export class Synchronizer {
  /** Socket connection */
  public socket: SocketIOClient.Socket
  /** Actions queued to send */
  public actionQueue: types.BaseAction[]
  /** Timestamped action log */
  public actionLog: types.BaseAction[]
  /** Middleware to use */
  public middleware: Middleware
  /** The address of the nodejs server that handles syncing */
  public syncAddress: string
  /** The function to call after state is synced with backend */
  public initStateCallback: (state: State) => void

  /* Make sure Session state is loaded before initializing this class */
  constructor (initialJson: State, syncAddress: string,
               initStateCallback: (state: State) => void) {
    this.syncAddress = syncAddress
    this.initStateCallback = initStateCallback

    this.actionQueue = []
    this.actionLog = []

    const self = this

    /* sync every time an action is dispatched */
    this.middleware = () => (
      next: Dispatch
    ) => (action) => {
      /* Only send back actions that originated locally */
      if (Session.id === action.sessionId) {
        self.actionQueue.push(action)
        self.sendActions()
      }
      return next(action)
    }

    const socket = io.connect(this.syncAddress)
    this.socket = socket

    this.socket.on(EventName.CONNECT, () => {
      /* Send the registration message to the backend */
      // If the store has been initialized, use it
      let sessionState = Session.getState()
      if (sessionState.session.id.length === 0) {
        // Otherwise use the loaded json
        sessionState = initialJson
      }
      self.socket.emit(EventName.REGISTER, sessionState)
      Session.updateStatusDisplay(ConnectionStatus.UNSAVED)
    })

    /* on receipt of registration ack from backend
       init synced state then send any queued actions */
    this.socket.on(EventName.REGISTER_ACK, (syncState: State) => {
      self.initStateCallback(syncState)
      self.sendActions()
    })

    this.socket.on(EventName.ACTION_BROADCAST, (action: types.ActionType) => {
      // actionLog matches backend action ordering
      self.actionLog.push(action)
      // Dispatch any actions broadcasted from other sessions
      if (action.sessionId !== Session.id) {
        Session.dispatch(action)
      }
    })

    // If backend disconnects, keep trying to reconnect
    this.socket.on(EventName.DISCONNECT, () => {
      Session.updateStatusDisplay(ConnectionStatus.RECONNECTING)
      // On reconnect, just update store instead of re-initializing it
      self.initStateCallback = (state: State) => {
        Session.dispatch(updateTask(state.task))
      }
    })
  }

  /**
   * Send all queued actions to the backend
   */
  public sendActions () {
    if (this.socket.connected) {
      if (this.actionQueue.length > 0) {
        this.socket.emit(
        EventName.ACTION_SEND, JSON.stringify({
          id: Session.getState().task.config.taskId,
          project: Session.getState().task.config.projectName,
          worker: Session.getState().user.id,
          actions: this.actionQueue
        }))
        this.actionQueue = []
      }
    }
  }
}

export default Synchronizer

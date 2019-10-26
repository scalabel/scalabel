import _ from 'lodash'
import { Dispatch, Middleware } from 'redux'
import io from 'socket.io-client'
import { updateTask } from '../action/common'
import * as types from '../action/types'
import { State } from '../functional/types'
import { EventName } from '../server/types'
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
  /** The function to call after state is synced with backend */
  public initStateCallback: (state: State) => void

  /* Make sure Session state is loaded before initializing this class */
  constructor (
    taskIndex: number, projectName: string,
    initStateCallback: (state: State) => void) {
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
        if (Session.autosave) {
          self.sendActions()
        }
      }
      return next(action)
    }

    // use the same port as http
    const syncAddress = 'http://localhost:' + location.port
    const socket = io.connect(syncAddress)
    this.socket = socket

    this.socket.on(EventName.CONNECT, () => {
      /* Send the registration message to the backend */
      self.socket.emit(
        EventName.REGISTER, JSON.stringify({
          project: projectName,
          index: taskIndex,
          sessId: Session.id
        }))
      Session.updateStatus(ConnectionStatus.UNSAVED)
    })

    /* on receipt of registration back from backend
       init synced state then send any queued actions */
    this.socket.on(EventName.REGISTER_ACK, (syncState: State) => {
      self.initStateCallback(syncState)
      self.sendActions()
    })

    this.socket.on(EventName.ACTION_BROADCAST, (action: types.ActionType) => {
      // actionLog matches backend action ordering
      self.actionLog.push(action)
      if (types.TASK_ACTION_TYPES.includes(action.type)) {
        if (action.sessionId !== Session.id) {
          // Dispatch any task actions broadcasted from other sessions
          Session.dispatch(action)
        } else {
          // Otherwise, indicate that task action from this session was saved
          Session.updateStatus(ConnectionStatus.SAVED)
          setTimeout(() => {
            Session.updateStatus(ConnectionStatus.UNSAVED)
          }, 5000)
        }
      }
    })

    // If backend disconnects, keep trying to reconnect
    this.socket.on(EventName.DISCONNECT, () => {
      Session.updateStatus(ConnectionStatus.RECONNECTING)
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
        const taskActions = this.actionQueue.filter((action) => {
          return types.TASK_ACTION_TYPES.includes(action.type)
        })
        if (taskActions.length > 0) {
          Session.updateStatus(ConnectionStatus.SAVING)
        }

        const sessionState = Session.getState()
        this.socket.emit(
          EventName.ACTION_SEND, JSON.stringify({
            taskId: sessionState.task.config.taskId,
            project: sessionState.task.config.projectName,
            sessId: sessionState.session.id,
            actions: this.actionQueue
          }))
        this.actionQueue = []
      }
    }
  }
}

export default Synchronizer

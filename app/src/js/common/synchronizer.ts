import _ from 'lodash'
import { Dispatch, Middleware } from 'redux'
import io from 'socket.io-client'
import { updateTask } from '../action/common'
import * as types from '../action/types'
import { State } from '../functional/types'
import { EventName, RegisterMessageType, SyncActionMessageType } from '../server/types'
import Session, { ConnectionStatus } from './session'

const CONFIRMATION_MESSAGE =
  'You have unsaved changes that will be lost if you leave this page. '

/**
 * Synchronizes data with other sessions
 */
export class Synchronizer {
  /** Socket connection */
  public socket: SocketIOClient.Socket
  /** Actions queued to send */
  public actionQueue: types.BaseAction[]
  /** Actions in process of being saved */
  public saveActions: types.BaseAction[][]
  /** Timestamped action log */
  public actionLog: types.BaseAction[]
  /** Middleware to use */
  public middleware: Middleware
  /** The function to call after state is synced with backend */
  public initStateCallback: (state: State) => void
  /** The user/browser id, constant across sessions */
  public userId: string

  /* Make sure Session state is loaded before initializing this class */
  constructor (
    taskIndex: number, projectName: string, userId: string,
    initStateCallback: (state: State) => void) {
    this.initStateCallback = initStateCallback

    this.actionQueue = []
    this.saveActions = []
    this.actionLog = []
    this.userId = userId

    const self = this

    /* sync every time an action is dispatched */
    this.middleware = () => (
      next: Dispatch
    ) => (action) => {
      /* Only send back actions that originated locally */
      action.userId = this.userId
      if (Session.id === action.sessionId) {
        self.actionQueue.push(action)
        if (Session.autosave) {
          self.sendQueuedActions()
        } else if (types.TASK_ACTION_TYPES.includes(action.type) &&
          Session.status !== ConnectionStatus.RECONNECTING &&
          Session.status !== ConnectionStatus.SAVING) {
          Session.updateStatus(ConnectionStatus.UNSAVED)
        }
      }
      return next(action)
    }

    // use the same address as http
    const syncAddress = location.origin
    const socket = io.connect(
      syncAddress,
      { transports: ['websocket'], upgrade: false }
    )
    this.socket = socket

    this.socket.on(EventName.CONNECT, () => {
      const message: RegisterMessageType = {
        projectName,
        taskIndex,
        sessionId: Session.id,
        userId: this.userId
      }
      /* Send the registration message to the backend */
      self.socket.emit(EventName.REGISTER, JSON.stringify(message))
      Session.updateStatus(ConnectionStatus.UNSAVED)
    })

    /* on receipt of registration back from backend
       init synced state then send any queued actions */
    this.socket.on(EventName.REGISTER_ACK, (syncState: State) => {
      self.initStateCallback(syncState)
      for (const saveActionList of this.saveActions) {
        self.sendActions(saveActionList)
      }
      if (Session.autosave) {
        self.sendQueuedActions()
      }
    })

    this.socket.on(
      EventName.ACTION_BROADCAST, (actionList: types.ActionType[]) => {
        // can remove 1st set of stored actions when saving is done
        this.saveActions.shift()
        for (const action of actionList) {
          // actionLog matches backend action ordering
          self.actionLog.push(action)
          if (types.TASK_ACTION_TYPES.includes(action.type)) {
            if (action.sessionId !== Session.id) {
              // Dispatch any task actions broadcasted from other sessions
              Session.dispatch(action)
            } else {
              // Otherwise, ack indicates successful save
              Session.updateStatus(ConnectionStatus.NOTIFY_SAVED)
              this.timeoutUpdateStatus(ConnectionStatus.SAVED, 5)
            }
          }
        }
      })

    // If backend disconnects, keep trying to reconnect
    this.socket.on(EventName.DISCONNECT, () => {
      Session.updateStatus(ConnectionStatus.RECONNECTING)
      if (Session.autosave) {
        // On reconnect, just update store instead of re-initializing it
        self.initStateCallback = (state: State) => {
          Session.dispatch(updateTask(state.task))
        }
      } else {
        // With manual saving, keep unsaved changes after reconnect
        self.initStateCallback = () => { return }
      }
    })

    // Add pop up to warn user when leaving with unsaved changes
    window.onbeforeunload = (e: BeforeUnloadEvent) => {
      if (
        Session.status === ConnectionStatus.RECONNECTING ||
        Session.status === ConnectionStatus.SAVING ||
        Session.status === ConnectionStatus.UNSAVED
      ) {
        e.returnValue = CONFIRMATION_MESSAGE // Gecko + IE
        return CONFIRMATION_MESSAGE // Gecko + Webkit, Safari, Chrome etc.
      }
    }
  }

  /**
   * Send all queued actions to the backend
   */
  public sendQueuedActions () {
    if (this.socket.connected) {
      if (this.actionQueue.length > 0) {
        this.saveActions.push(this.actionQueue)
        this.sendActions(this.actionQueue)
        this.actionQueue = []
      }
    }
  }

  /**
   * Send given actions to the backend
   */
  public sendActions (actionList: types.BaseAction[]) {
    const sessionState = Session.getState()
    const message: SyncActionMessageType = {
      taskId: sessionState.task.config.taskId,
      projectName: sessionState.task.config.projectName,
      sessionId: sessionState.session.id,
      actions: actionList
    }
    this.socket.emit(EventName.ACTION_SEND, JSON.stringify(message))
    Session.updateStatus(ConnectionStatus.SAVING)
  }

  /**
   * Update status after timeout if status hasn't changed since then
   */
  public timeoutUpdateStatus (newStatus: ConnectionStatus, seconds: number) {
    const statusChangeCountBefore = Session.statusChangeCount
    setTimeout(() => {
      // don't update if other effect occurred in between
      if (Session.statusChangeCount === statusChangeCountBefore) {
        Session.updateStatus(newStatus)
      }
    }, seconds * 1000)
  }
}

export default Synchronizer

import _ from 'lodash'
import { Dispatch, Middleware } from 'redux'
import io from 'socket.io-client'
import uuid4 from 'uuid/v4'
import { updateTask } from '../action/common'
import * as types from '../action/types'
import { State } from '../functional/types'
import { ActionPacketType, EventName, RegisterMessageType,
  SyncActionMessageType } from '../server/types'
import Session, { ConnectionStatus } from './session'

const CONFIRMATION_MESSAGE =
  'You have unsaved changes that will be lost if you leave this page. '

/**
 * Synchronizes data with other sessions
 */
export class Synchronizer {
  /** Socket connection */
  public socket: SocketIOClient.Socket
  /** Actions queued to be sent to the backend */
  public actionQueue: types.BaseAction[]
  /** Actions in the process of being saved, mapped by packet id */
  public actionsToSave: { [id: string]: ActionPacketType }
  /** Timestamped log for completed actions */
  public actionLog: types.BaseAction[]
  /** The function to call after state is synced with backend */
  public initStateCallback: (state: State) => void
  /** Name of the project */
  public projectName: string
  /** Index of the task */
  public taskIndex: number
  /** Middleware executed on action dispatch */
  public middleware: Middleware
  /** The user/browser id, constant across sessions */
  public userId: string

  /* Make sure Session state is loaded before initializing this class */
  constructor (
    taskIndex: number, projectName: string, userId: string,
    initStateCallback: (state: State) => void) {
    this.taskIndex = taskIndex
    this.projectName = projectName
    this.initStateCallback = initStateCallback

    this.actionQueue = []
    this.actionsToSave = {}
    this.actionLog = []
    this.userId = userId

    // use the same address as http
    const syncAddress = location.origin
    const socket = io.connect(
      syncAddress,
      { transports: ['websocket'], upgrade: false }
    )
    this.socket = socket

    this.socket.on(EventName.CONNECT, this.connectHandler.bind(this))
    this.socket.on(EventName.REGISTER_ACK, this.registerAckHandler.bind(this))
    this.socket.on(EventName.ACTION_BROADCAST,
      this.actionBroadcastHandler.bind(this))
    this.socket.on(EventName.DISCONNECT, this.disconnectHandler.bind(this))
    window.onbeforeunload = this.warningPopup.bind(this)

    /* Called every time an action is dispatched to the session */
    const self = this
    this.middleware = () => (
      next: Dispatch
    ) => (action: types.BaseAction) => {
      action.userId = this.userId
      /* Only send back actions that originated locally */
      if (Session.id === action.sessionId && !action.frontendOnly) {
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
  }

  /**
   * Displays pop-up warning user when leaving with unsaved changes
   */
  public warningPopup (e: BeforeUnloadEvent) {
    if (
      Session.status === ConnectionStatus.RECONNECTING ||
      Session.status === ConnectionStatus.SAVING ||
      Session.status === ConnectionStatus.UNSAVED
    ) {
      e.returnValue = CONFIRMATION_MESSAGE // Gecko + IE
      return CONFIRMATION_MESSAGE // Gecko + Webkit, Safari, Chrome etc.
    }
  }

  /**
   * Called when io socket establishes a connection
   * Registers the session with the backend, triggering a register ack
   */
  public connectHandler () {
    const message: RegisterMessageType = {
      projectName: this.projectName,
      taskIndex: this.taskIndex,
      sessionId: Session.id,
      userId: this.userId
    }
    /* Send the registration message to the backend */
    this.socket.emit(EventName.REGISTER, JSON.stringify(message))
    Session.updateStatus(ConnectionStatus.UNSAVED)
  }

  /**
   * Called when backend sends ack of registration of this session
   * Initialized synced state, and sends any queued actions
   */
  public registerAckHandler (syncState: State) {
    this.initStateCallback(syncState)
    for (const actionPacket of this.listActionPackets()) {
      this.sendActions(actionPacket)
    }
    if (Session.autosave) {
      this.sendQueuedActions()
    }
  }

  /**
   * Called when backend sends ack for actions that were sent to be synced
   * Updates relevant queues and syncs actions from other sessions
   */
  public actionBroadcastHandler (actionPacket: ActionPacketType) {
    // remove stored actions when they are acked
    if (actionPacket.id in this.actionsToSave) {
      delete this.actionsToSave[actionPacket.id]
    }

    for (const action of actionPacket.actions) {
      // actionLog matches backend action ordering
      this.actionLog.push(action)
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
  }

  /**
   * Called when session disconnects from backend
   * Prepares for reconnect by updating initial callback
   */
  public disconnectHandler () {
    Session.updateStatus(ConnectionStatus.RECONNECTING)
    if (Session.autosave) {
      this.initStateCallback = this.autosaveReconnectCallback
    } else {
      // With manual saving, keep unsaved changes after reconnect
      this.initStateCallback = () => { return }
    }
  }

  /**
   * Called when session reconnects (with autosave)
   */
  public autosaveReconnectCallback (state: State) {
    // updateTask is not a task action, so will not sync again
    Session.dispatch(updateTask(state.task))

    // re-apply frontend task actions after updating task from backend
    for (const actionPacket of this.listActionPackets()) {
      for (const action of actionPacket.actions) {
        if (types.TASK_ACTION_TYPES.includes(action.type)) {
          action.frontendOnly = true
          Session.dispatch(action)
        }
      }
    }
  }

  /**
   * Converts dict of action packets to a list
   */
  public listActionPackets (): ActionPacketType[] {
    return Object.keys(this.actionsToSave).map(
      (key) => this.actionsToSave[key])
  }

  /**
   * Send all queued actions to the backend
   * Should only call this once per action, since id shouldn't change
   */
  public sendQueuedActions () {
    if (this.socket.connected) {
      if (this.actionQueue.length > 0) {
        const packetId = uuid4()
        const actionPacket: ActionPacketType = {
          actions: this.actionQueue,
          id: packetId
        }
        this.actionsToSave[packetId] = actionPacket
        this.sendActions(actionPacket)
        this.actionQueue = []
      }
    }
  }

  /**
   * Send given action packet to the backend
   * Can be called multiple times if previous attempts aren't acked
   */
  public sendActions (actionPacket: ActionPacketType) {
    const sessionState = Session.getState()
    const message: SyncActionMessageType = {
      taskId: sessionState.task.config.taskId,
      projectName: sessionState.task.config.projectName,
      sessionId: sessionState.session.id,
      actions: actionPacket
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
      // Don't update if other effect occurred in between
      if (Session.statusChangeCount === statusChangeCountBefore) {
        Session.updateStatus(newStatus)
      }
    }, seconds * 1000)
  }
}

export default Synchronizer

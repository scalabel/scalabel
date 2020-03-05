import { Store } from 'redux'
import { StateWithHistory } from 'redux-undo'
import io from 'socket.io-client'
import { sprintf } from 'sprintf-js'
import uuid4 from 'uuid/v4'
import { BaseAction } from '../action/types'
import { configureStore } from '../common/configure_store'
import { State } from '../functional/types'
import {
  EventName, RegisterMessageType, SyncActionMessageType
} from '../server/types'
import Logger from './logger'

/**
 * Watches and modifies state based on what user sessions do
 */
export class VirtualSession {
  /** bot user id */
  public userId: string
  /** virtual session id */
  public sessionId: string
  /** name of the project */
  public projectName: string
  /** index of the task */
  public taskIndex: number
  /** the address of the io server */
  public address: string
  /** Number of actions received via broadcast */
  public actionCount: number
  /** The store to save state */
  protected store: Store<StateWithHistory<State>>
  /** Socket connection */
  protected socket: SocketIOClient.Socket
  /** Timestamped log for completed actions */
  protected actionLog: BaseAction[]
  /** Log of packets that have been acked */
  protected ackedPackets: Set<string>

  constructor (
    userId: string, address: string, projectName: string, taskIndex: number) {
    this.userId = userId
    this.address = address
    this.projectName = projectName
    this.taskIndex = taskIndex
    this.sessionId = uuid4()
    this.actionCount = 0

    // create a socketio client
    const socket = io.connect(
      this.address,
      { transports: ['websocket'], upgrade: false }
    )
    this.socket = socket

    this.socket.on(EventName.CONNECT, this.connectHandler.bind(this))
    this.socket.on(EventName.REGISTER_ACK, this.registerAckHandler.bind(this))
    this.socket.on(EventName.ACTION_BROADCAST,
      this.actionBroadcastHandler.bind(this))

    this.store = configureStore({})

    this.actionLog = []
    this.ackedPackets = new Set()
  }

  /**
   * Called when io socket establishes a connection
   * Registers the session with the backend, triggering a register ack
   */
  public connectHandler () {
    const message: RegisterMessageType = {
      projectName: this.projectName,
      taskIndex: this.taskIndex,
      sessionId: this.sessionId,
      userId: this.userId,
      address: this.address,
      bot: true
    }
    /* Send the registration message to the backend */
    this.socket.emit(EventName.REGISTER, message)
  }

  /**
   * Called when backend sends ack of registration of this session
   * Initialized synced state
   */
  public registerAckHandler (syncState: State) {
    this.store = configureStore(syncState)
  }

  /**
   * Called when backend sends ack for actions that were sent to be synced
   * Simply logs these actions for now
   */
  public actionBroadcastHandler (
    message: SyncActionMessageType) {
    const actionPacket = message.actions
    // if action was already acked, ignore it
    if (this.ackedPackets.has(actionPacket.id)) {
      return
    }
    this.ackedPackets.add(actionPacket.id)

    for (const action of actionPacket.actions) {
      this.actionCount += 1
      this.actionLog.push(action)
      Logger.info(
        sprintf('Virtual session received action of type %s', action.type))
    }
  }

  /**
   * Close any external resources
   */
  public kill () {
    this.socket.disconnect()
  }

}

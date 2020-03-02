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
  /** virtual session id */
  public sessionId: string
  /** The store to save states */
  protected store: Store<StateWithHistory<State>>
  /** Socket connection */
  protected socket: SocketIOClient.Socket
  /** name of the project */
  protected projectName: string
  /** id of the task */
  protected taskIndex: number
  /** virutal user id */
  protected userId: string
  /** Timestamped log for completed actions */
  protected actionLog: BaseAction[]
  /** Log of packets that have been acked */
  protected ackedPackets: Set<string>

  constructor (projectName: string, taskIndex: number, address: string) {
    this.projectName = projectName
    this.taskIndex = taskIndex
    this.sessionId = uuid4()
    this.userId = ''

    // create a socketio client
    const socket = io.connect(
      address,
      { transports: ['websocket'], upgrade: false }
    )
    this.socket = socket
    this.store = configureStore({})

    this.actionLog = []
    this.ackedPackets = new Set()
  }

  /**
   * Add handlers to socket
   */
  public listen () {
    this.socket.on(EventName.CONNECT, this.connectHandler.bind(this))
    this.socket.on(EventName.REGISTER_ACK, this.registerAckHandler.bind(this))
    this.socket.on(EventName.ACTION_BROADCAST,
      this.actionBroadcastHandler.bind(this))
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
      address: ''
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
      this.actionLog.push(action)
      Logger.info(
        sprintf('Virtual session for \
project "%s", task %d \
received action of type %s',
        this.projectName, this.taskIndex, action.type))
    }
  }
}

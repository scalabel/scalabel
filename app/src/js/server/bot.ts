import { Store } from 'redux'
import { StateWithHistory } from 'redux-undo'
import io from 'socket.io-client'
import { sprintf } from 'sprintf-js'
import uuid4 from 'uuid/v4'
import { BaseAction } from '../action/types'
import { configureStore } from '../common/configure_store'
import { State } from '../functional/types'
import {
  BotData, EventName, RegisterMessageType, SyncActionMessageType
} from '../server/types'
import Logger from './logger'

/**
 * Manages virtual sessions for a single bot
 */
export class Bot {
  /** project name */
  public projectName: string
  /** task index */
  public taskIndex: number
  /** bot user id */
  public botId: string
  /** an arbitrary session id */
  public sessionId: string
  /** address for session connections */
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

  constructor (botData: BotData) {
    this.projectName = botData.projectName
    this.taskIndex = botData.taskIndex
    this.botId = botData.botId
    this.address = botData.address
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
      userId: this.botId,
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

  /**
   * Gets the number of actions for the bot
   */
  public getActionCount (): number {
    return this.actionCount
  }

  /**
   * Sets action counts to 0 for the bot
   */

  public resetActionCount () {
    this.actionCount = 0
  }

  /**
   * Wraps instance variables into data object
   */
  public getData (): BotData {
    return {
      botId: this.botId,
      projectName: this.projectName,
      taskIndex: this.taskIndex,
      address: this.address
    }
  }
}

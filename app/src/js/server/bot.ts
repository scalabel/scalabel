import axios, { AxiosRequestConfig } from 'axios'
import { Store } from 'redux'
import { StateWithHistory } from 'redux-undo'
import io from 'socket.io-client'
import { sprintf } from 'sprintf-js'
import uuid4 from 'uuid/v4'
import { ADD_LABELS, AddLabelsAction, BaseAction } from '../action/types'
import { configureStore } from '../common/configure_store'
import { ShapeTypeName } from '../common/types'
import { PolygonType, RectType, State } from '../functional/types'
import Logger from './logger'
import { ModelInterface } from './model_interface'
import {
  ActionPacketType, BotData, EventName,
  ModelQuery, RegisterMessageType, SyncActionMessageType
} from './types'
import { getPyConnFailedMsg, index2str } from './util'

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
  /** address of model server */
  protected modelAddress: URL
  /** interface with model data type */
  protected modelInterface: ModelInterface
  /** the axios http config */
  protected axiosConfig: AxiosRequestConfig

  constructor (botData: BotData, botHost: string, botPort: number) {
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

    this.modelAddress = new URL(botHost)
    this.modelAddress.port = botPort.toString()

    this.modelInterface = new ModelInterface(this.projectName, this.sessionId)

    this.axiosConfig = {
      headers: {
        'Content-Type': 'application/json'
      }
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
  public async actionBroadcastHandler (
    message: SyncActionMessageType) {
    const actionPacket = message.actions
    // if action was already acked, or if action came from a bot, ignore it
    if (this.ackedPackets.has(actionPacket.id)
      || message.bot
      || message.sessionId === this.sessionId) {
      return
    }

    this.ackedPackets.add(actionPacket.id)

    // precompute queries so they can potentially execute in parallel
    const queries = this.packetToQueries(actionPacket)
    const actions = await this.executeQueries(queries)
    if (actions.length > 0) {
      this.broadcastActions(actions)
    }
  }

  /**
   * Broadcast the synthetically generated actions
   */
  public broadcastActions (actions: AddLabelsAction[]) {
    const actionPacket: ActionPacketType = {
      actions,
      id: uuid4()
    }
    const message: SyncActionMessageType = {
      taskId: index2str(this.taskIndex),
      projectName: this.projectName,
      sessionId: this.sessionId,
      actions: actionPacket,
      bot: true
    }
    this.socket.emit(EventName.ACTION_SEND, message)
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

  /**
   * Execute queries and get the resulting actions
   */
  private async executeQueries (
    queries: ModelQuery[]): Promise<AddLabelsAction[]> {
    const actions: AddLabelsAction[] = []
    const modelEndpoint = new URL(queries[0].endpoint, this.modelAddress)
    const itemIndex = queries[0].itemIndex
    const allData = []
    for (const query of queries) {
      const data = query.data
      allData.push(data)
    }
      // const modelEndpoint = new URL(query.endpoint, this.modelAddress)
    try {
      const response = await axios.post(
        modelEndpoint.toString(), allData, this.axiosConfig
      )
      const data: number[][][] = response.data.points
      for (const datum of data) {
        const action = this.modelInterface.makePolyAction(
          datum, itemIndex
        )
        actions.push(action)
      }
      // Logger.info(sprintf('Got a %s response from the model with data: %s',
      //   response.status.toString(), response.data.points))
    } catch (e) {
      Logger.info(getPyConnFailedMsg(modelEndpoint.toString(), e.message))
    }
    return actions
  }

  /**
   * Compute queries for the actions in the packet
   */
  private packetToQueries (packet: ActionPacketType): ModelQuery[] {
    const queries: ModelQuery[] = []
    for (const action of packet.actions) {
      if (action.sessionId !== this.sessionId) {
        this.actionCount += 1
        this.actionLog.push(action)
        this.store.dispatch(action)
        Logger.info(
          sprintf('Bot received action of type %s', action.type))

        const state = this.store.getState().present
        if (action.type === ADD_LABELS) {
          const query = this.actionToQuery(
            state, action as AddLabelsAction)
          if (query) {
            queries.push(query)
          }
        }
      }
    }
    return queries
  }

  /**
   * Generate BDD data format item corresponding to the action
   * Only handles box2d/polygon2d actions, so assume a single label/shape/item
   */
  private actionToQuery (
    state: State, action: AddLabelsAction): ModelQuery | null {
    const shapeType = action.shapeTypes[0][0][0]
    const shape = action.shapes[0][0][0]
    const labelType = action.labels[0][0].type
    const itemIndex = action.itemIndices[0]
    const item = state.task.items[itemIndex]
    const url = Object.values(item.urls)[0]

    switch (shapeType) {
      case ShapeTypeName.RECT:
        return this.modelInterface.makeRectQuery(
          shape as RectType, url, itemIndex
        )
      case ShapeTypeName.POLYGON_2D:
        return this.modelInterface.makePolyQuery(
          shape as PolygonType, url, itemIndex, labelType
        )
      default:
        return null
    }
  }
}

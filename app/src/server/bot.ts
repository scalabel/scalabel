import io from "socket.io-client"

import { configureStore } from "../common/configure_store"
import { uid } from "../common/uid"
import { index2str } from "../common/util"
import { ADD_LABELS, PREDICT } from "../const/action"
import { EventName, RedisChannel } from "../const/connection"
import {
  AddLabelsAction,
  BaseAction,
  PredictionAction,
  UpdateModelStatusAction
} from "../types/action"
import { RedisConfig } from "../types/config"
import {
  ActionPacketType,
  BotData,
  ModelStatusMessageType,
  ModelRegisterMessageType,
  ModelRequest,
  ModelRequestMessageType,
  RegisterMessageType,
  SyncActionMessageType
} from "../types/message"
import { ReduxStore } from "../types/redux"
import { IntrinsicsType, ModelStatus, RectType, State } from "../types/state"
import Logger from "./logger"
import { ModelInterface } from "./model_interface"
import { makeRedisPubSub, RedisPubSub } from "./redis_pub_sub"
import {
  ConnectionRequestMode,
  LabelTypeName,
  ModelRequestMode,
  ShapeTypeName
} from "../const/common"
import { updateModelStatus } from "../action/common"

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
  /** label type */
  public labelType: string
  /** The store to save state */
  protected store: ReduxStore
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
  /** the Redis message broker to publish messages */
  protected publisher: RedisPubSub
  /** the Redis message broker to subscribe to model server responses */
  protected responseSubscriber: RedisPubSub
  /** the Redis message broker to subscribe to connection notifications */
  protected notifySubscriber: RedisPubSub
  /** the unique id that is used for identification with the model server */
  private readonly clientId: string
  /** the Redis channel to send task requests */
  private requestsChannel: string
  /** Number of actions received via broadcast */
  private actionCount: number

  /**
   * Constructor
   *
   * @param botData
   * @param botHost
   * @param botPort
   * @param redisConfig
   */
  constructor(
    botData: BotData,
    botHost: string,
    botPort: number,
    redisConfig: RedisConfig
  ) {
    this.projectName = botData.projectName
    this.taskIndex = botData.taskIndex
    this.botId = botData.botId
    this.address = botData.address
    this.labelType = botData.labelType
    this.sessionId = uid()

    this.publisher = makeRedisPubSub(redisConfig)
    this.responseSubscriber = makeRedisPubSub(redisConfig)
    this.notifySubscriber = makeRedisPubSub(redisConfig)

    this.actionCount = 0

    // Create a socketio client
    const socket = io.connect(this.address, {
      transports: ["websocket"],
      upgrade: false
    })
    this.socket = socket

    this.socket.on(EventName.CONNECT, this.connectHandler.bind(this))
    this.socket.on(EventName.REGISTER_ACK, this.registerAckHandler.bind(this))
    this.socket.on(
      EventName.ACTION_BROADCAST,
      this.actionBroadcastHandler.bind(this)
    )

    this.store = configureStore({})

    this.actionLog = []
    this.ackedPackets = new Set()

    this.modelAddress = new URL(botHost)
    this.modelAddress.port = botPort.toString()

    this.modelInterface = new ModelInterface(this.projectName, this.sessionId)

    this.clientId = `${this.projectName}_${index2str(this.taskIndex)}`
    this.requestsChannel = RedisChannel.MODEL_REQUEST
  }

  /**
   * Listen for model response
   */
  public async listen(): Promise<void> {
    const responseChannel = `${RedisChannel.MODEL_RESPONSE}_${this.clientId}`
    const notifyChannel = `${RedisChannel.MODEL_NOTIFY}_${this.clientId}`
    await this.responseSubscriber.subscribeEvent(
      responseChannel,
      this.modelResponseHandler.bind(this)
    )
    await this.notifySubscriber.subscribeEvent(
      notifyChannel,
      this.modelNotificationHandler.bind(this)
    )
  }

  /**
   * Called when io socket establishes a connection
   * Registers the session with the backend, triggering a register ack
   */
  public connectHandler(): void {
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
   *
   * @param syncState
   */
  public registerAckHandler(syncState: State): void {
    this.store = configureStore(syncState)

    const modelRegisterMessage: ModelRegisterMessageType = {
      clientId: this.clientId,
      handshakeChannel: `${RedisChannel.MODEL_NOTIFY}_${this.clientId}`,
      request: ConnectionRequestMode.CONNECT
    }
    this.publisher.publishEvent(
      RedisChannel.MODEL_REGISTER,
      modelRegisterMessage
    )
  }

  /**
   * Called when backend sends ack for actions that were sent to be synced
   * Simply logs these actions for now
   *
   * @param message
   */
  public async actionBroadcastHandler(
    message: SyncActionMessageType
  ): Promise<void> {
    const actionPacket = message.actions
    // If action was already acked, or if action came from a bot, ignore it
    if (
      this.ackedPackets.has(actionPacket.id) ||
      message.bot ||
      message.sessionId === this.sessionId
    ) {
      return
    }

    this.ackedPackets.add(actionPacket.id)

    const modelRequests = this.packetToRequests(actionPacket)
    this.executeRequests(modelRequests, actionPacket.id)
  }

  /**
   * Broadcast the synthetically generated actions
   *
   * @param actions
   * @param triggerId
   */
  public broadcastActions(actions: BaseAction[], triggerId?: string): void {
    const actionPacket: ActionPacketType = {
      actions,
      id: uid(),
      triggerId
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
   *
   * @param active
   */
  public setActivate(active: boolean): void {
    const modelStatusMessage: ModelStatusMessageType = {
      projectName: this.projectName,
      taskId: index2str(this.taskIndex),
      active
    }
    this.publisher.publishEvent(RedisChannel.MODEL_STATUS, modelStatusMessage)
    if (!active) {
      this.socket.disconnect()
    }
  }

  /**
   * Gets the number of actions for the bot
   */
  public getActionCount(): number {
    return this.actionCount
  }

  /**
   * Sets action counts to 0 for the bot
   */
  public resetActionCount(): void {
    this.actionCount = 0
  }

  /**
   * Wraps instance variables into data object
   */
  public getData(): BotData {
    return {
      botId: this.botId,
      projectName: this.projectName,
      taskIndex: this.taskIndex,
      address: this.address,
      labelType: this.labelType
    }
  }

  /**
   * Get the current redux state
   */
  public getState(): State {
    return this.store.getState().present
  }

  /**
   * Execute requests and get the resulting actions
   *
   * @param modelRequests
   * @param actionPacketId
   */
  private executeRequests(
    modelRequests: Array<ModelRequest | null>,
    actionPacketId: string
  ): void {
    const items = []
    const itemIndices = []
    for (const modelRequest of modelRequests) {
      if (modelRequest !== null) {
        items.push(modelRequest.data[0])
        itemIndices.push(modelRequest.itemIndices[0])
      }
    }
    try {
      const modelRequestMessage: ModelRequestMessageType = {
        clientId: this.clientId,
        mode: ModelRequestMode.INFERENCE,
        taskType: this.labelType,
        projectName: this.projectName,
        taskId: index2str(this.taskIndex),
        items,
        itemIndices,
        dataSize: items.length,
        actionPacketId,
        channel:
          RedisChannel.MODEL_RESPONSE +
          "_" +
          this.projectName +
          "_" +
          index2str(this.taskIndex)
      }
      this.publisher.publishEvent(this.requestsChannel, modelRequestMessage)
    } catch (e) {
      Logger.info("Failed!")
    }
  }

  /**
   * Compute requests for the actions in the packet
   *
   * @param packet
   */
  private packetToRequests(
    packet: ActionPacketType
  ): Array<ModelRequest | null> {
    const modelRequests: Array<ModelRequest | null> = []
    for (const action of packet.actions) {
      if (action.sessionId !== this.sessionId) {
        this.actionCount += 1
        this.actionLog.push(action)
        this.store.dispatch(action)
        Logger.info(`Bot received action of type ${action.type}`)

        const state = this.store.getState().present
        if (action.type === PREDICT) {
          const request = this.imageActionToRequest(
            state,
            action as PredictionAction
          )
          modelRequests.push(request)
        } else if (action.type === ADD_LABELS) {
          if (this.labelType === LabelTypeName.POLYGON_2D) {
            const request = this.boxActionToRequest(
              state,
              action as AddLabelsAction
            )
            if (request !== null) {
              modelRequests.push(request)
            }
          }
        }
      }
    }
    return modelRequests
  }

  /**
   * Generate BDD data format item corresponding to the action
   * Only handles box2d/polygon2d actions, so assume a single label/shape/item
   *
   * @param state
   * @param action
   */
  private imageActionToRequest(
    state: State,
    action: PredictionAction
  ): ModelRequest | null {
    const urls: string[] = []
    const intrinsics: Array<IntrinsicsType | undefined> = []
    for (const index of action.itemIndices) {
      const item = state.task.items[index]
      urls.push(Object.values(item.urls)[0])
      intrinsics.push(
        item.intrinsics != null ? Object.values(item.intrinsics)[0] : undefined
      )
    }
    return this.modelInterface.makeImageRequest(
      urls,
      action.itemIndices,
      intrinsics
    )
  }

  /**
   * Generate BDD data format item corresponding to the action
   * Only handles box2d/polygon2d actions, so assume a single label/shape/item
   *
   * @param state
   * @param action
   */
  private boxActionToRequest(
    state: State,
    action: AddLabelsAction
  ): ModelRequest | null {
    const shapeType = action.shapes[0][0][0].shapeType
    const shapes = action.shapes[0][0]
    const itemIndex = action.itemIndices[0]
    const item = state.task.items[itemIndex]
    const url = Object.values(item.urls)[0]

    if (shapeType === ShapeTypeName.RECT) {
      return this.modelInterface.makeRectRequest(
        shapes[0] as RectType,
        url,
        itemIndex
      )
    } else {
      return null
    }
  }

  /**
   * returned fields of model response
   *
   * @param _channel
   * @param modelResponse
   */
  private modelResponseHandler(_channel: string, modelResponse: string): void {
    const actions: AddLabelsAction[] = []

    const receivedData = JSON.parse(modelResponse)
    for (let index = 0; index < receivedData.output.length; index++) {
      const shapes: number[][] = receivedData.output[index].boxes
      const itemIndex: number = receivedData.itemIndices[index]
      const actionPacketId: string = receivedData.actionPacketId

      if (this.labelType === LabelTypeName.BOX_2D) {
        shapes.forEach((shape: number[]) => {
          const action = this.modelInterface.makeRectAction(shape, itemIndex)
          actions.push(action)
        })
      } else if (this.labelType === LabelTypeName.BOX_3D) {
        for (const box of shapes) {
          if (box[0] === 0 || box[1] === 0 || box[2] === 0) {
            Logger.info(`Ignoring box with 0 dimension: ${String(box)}`)
            continue
          } else {
            Logger.info(`Creating box: ${String(box)}`)
            const action = this.modelInterface.makeBox3dAction(box, itemIndex)
            actions.push(action)
          }
        }
      } else {
        const action = this.modelInterface.makePolyAction(shapes, itemIndex)
        actions.push(action)
      }

      // Dispatch the predicted actions locally
      for (const action of actions) {
        this.store.dispatch(action)
      }

      // Broadcast the predicted actions to other session
      if (actions.length > 0) {
        this.broadcastActions(actions, actionPacketId)
      }
    }
  }

  /**
   * returned fields of model response
   *
   * @param _channel
   * @param connectionResponse
   */
  private modelNotificationHandler(
    _channel: string,
    connectionResponse: string
  ): void {
    const receivedData = JSON.parse(connectionResponse)
    let action: UpdateModelStatusAction
    if (receivedData.status === "OK") {
      action = updateModelStatus(ModelStatus.READY)
      this.broadcastActions([action])
      this.requestsChannel = receivedData.requestsChannel
    } else {
      action = updateModelStatus(ModelStatus.INVALID)
      this.broadcastActions([action])
    }
  }
}

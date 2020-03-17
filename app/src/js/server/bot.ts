import axios, { AxiosRequestConfig } from 'axios'
import { Store } from 'redux'
import { StateWithHistory } from 'redux-undo'
import io from 'socket.io-client'
import { sprintf } from 'sprintf-js'
import uuid4 from 'uuid/v4'
import { addPolygon2dLabel } from '../action/polygon2d'
import { ADD_LABELS, AddLabelsAction, BaseAction } from '../action/types'
import { configureStore } from '../common/configure_store'
import { ShapeTypeName } from '../common/types'
import { PathPoint2D, PointType } from '../drawable/2d/path_point2d'
import { makeItemExport, makeLabelExport } from '../functional/states'
import { PolygonType, RectType, State } from '../functional/types'
import { polygonToExport } from '../server/export'
import {
  ActionPacketType, BotData, EventName, ModelEndpoint,
  ModelQuery, RegisterMessageType, SyncActionMessageType
} from '../server/types'
import Logger from './logger'
import { index2str } from './util'

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
  protected modelAddress: string

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

    // currently assume python is on the same server
    // should put the host/port in a config
    this.modelAddress = 'http://0.0.0.0:8080'
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
    // if action was already acked, ignore it
    if (this.ackedPackets.has(actionPacket.id)) {
      return
    }
    this.ackedPackets.add(actionPacket.id)

    // precompute queries so they can potentially execute in parallel
    let modelQueries: ModelQuery[] = []
    for (const action of actionPacket.actions) {
      if (action.sessionId !== this.sessionId) {
        this.actionCount += 1
        this.actionLog.push(action)
        this.store.dispatch(action)
        Logger.info(
          sprintf('Bot received action of type %s', action.type))

        const state = this.store.getState().present
        if (action.type === ADD_LABELS) {
          const actionQueries = this.getModelQueries(
            state, action as AddLabelsAction)
          modelQueries = modelQueries.concat(actionQueries)
        }
      }
    }

    // execute queries
    for (const modelQuery of modelQueries) {
      const data = modelQuery.itemExport
      const config: AxiosRequestConfig = {
        headers: {
          'Content-Type': 'application/json'
        }
      }
      const modelEndpoint = new URL(modelQuery.endpoint, this.modelAddress)
      try {
        Logger.info(sprintf('data: %s', data))
        const response = await axios.post(
          modelEndpoint.toString(), data, config
        )
        Logger.info('Got a response from the model')
        Logger.info(response.status.toString())
        Logger.info(response.data)

        const polyPoints = response.data.pred as number[][]
        const points = polyPoints.map((point: number[]) => {
          return (new PathPoint2D(
            point[0], point[1], PointType.VERTEX)).toPathPoint()
        })
        // const polygon: PolygonType = {
        //   points
        // }
        // TODO-mark label.manual as false
        // TODO- use changeShapes to broadcast in parallel
        // const refineAction = changeLabelShape(
        //   modelQuery.itemIndex, modelQuery.shapeId, polygon
        // )

        const refineAction = addPolygon2dLabel(
          modelQuery.itemIndex, -1, [0], points, true
        )
        refineAction.sessionId = this.sessionId

        // broadcast
        const actionPacketOut: ActionPacketType = {
          actions: [refineAction],
          id: uuid4()
        }
        const messageOut: SyncActionMessageType = {
          taskId: index2str(this.taskIndex),
          projectName: this.projectName,
          sessionId: this.sessionId,
          actions: actionPacketOut
        }
        this.socket.emit(EventName.ACTION_SEND, messageOut)
      } catch (e) {
        Logger.info(sprintf('Query to \"%s\" failed- make sure endpoint is \
correct and python server is running', modelEndpoint.toString()))
        Logger.info(e.message)
      }
    }
  }

  /**
   * Generate BDD data format item corresponding to the action
   */
  public getModelQueries (
    state: State, action: AddLabelsAction): ModelQuery[] {
    const queries: ModelQuery[] = []
    /* this only handles box2d and polygon2d actions,
     * so we can assume a single shape
     */
    for (const itemIndex of action.itemIndices) {
      const item = state.task.items[itemIndex]
      const url = Object.values(item.urls)[0]

      // TODO- don't just get first one
      const labelIndex = parseInt(Object.keys(item.labels)[0],10)
      const labelId = item.labels[labelIndex].id
      const shapeId = item.labels[labelId].shapes[0]

      const shapeType = action.shapeTypes[0][0][0]
      const shape = action.shapes[0][0][0]
      switch (shapeType) {
        case ShapeTypeName.RECT:
          const rectLabel = makeLabelExport({
            box2d: shape as RectType
          })
          const rectItem = makeItemExport({
            name: this.projectName,
            url,
            labels: [rectLabel]
          })
          const rectQuery = {
            itemExport: rectItem,
            endpoint: ModelEndpoint.POLYGON_RNN_BASE,
            shapeId,
            itemIndex
          }
          queries.push(rectQuery)
          break
        case ShapeTypeName.POLYGON_2D:
          const labelType = action.labels[0][0].type
          const poly2d = polygonToExport(shape as PolygonType, labelType)
          const polyLabel = makeLabelExport({
            poly2d
          })
          const polyItem = makeItemExport({
            name: this.projectName,
            url,
            labels: [polyLabel]
          })
          const polyQuery = {
            itemExport: polyItem,
            endpoint: ModelEndpoint.POLYGON_RNN_REFINE,
            shapeId,
            itemIndex
          }
          queries.push(polyQuery)
          break
        default:
          break
      }
    }

    return queries
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

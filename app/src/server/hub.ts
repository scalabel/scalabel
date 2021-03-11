import _ from "lodash"
import socketio from "socket.io"

import { index2str } from "../common/util"
import * as actionConsts from "../const/action"
import { EventName } from "../const/connection"
import { ServerConfig } from "../types/config"
import { RegisterMessageType, SyncActionMessageType } from "../types/message"
import { StateMetadata } from "../types/project"
import Logger from "./logger"
import * as path from "./path"
import { ProjectStore } from "./project_store"
import { RedisPubSub } from "./redis_pub_sub"
import { SocketServer } from "./socket_interface"
import { UserManager } from "./user_manager"
import { initSessionId, updateStateTimestamp } from "./util"

/**
 * Wraps socket.io handlers for saving, loading, and synchronization
 */
export class Hub {
  /** flag for sync */
  protected sync: boolean
  /** flag for bots */
  protected bots: boolean
  /** flag for autosave */
  protected autosave: boolean
  /** the project store */
  protected projectStore: ProjectStore
  /** the user manager */
  protected userManager: UserManager
  /** the redis message broker */
  protected publisher: RedisPubSub

  /**
   * Constructor
   *
   * @param config
   * @param projectStore
   * @param userManager
   * @param publisher
   */
  constructor(
    config: ServerConfig,
    projectStore: ProjectStore,
    userManager: UserManager,
    publisher: RedisPubSub
  ) {
    this.sync = config.mode.sync
    this.autosave = config.mode.autosave
    this.bots = config.bot.on
    this.projectStore = projectStore
    this.userManager = userManager
    this.publisher = publisher
  }

  /**
   * Listens for websocket connections
   *
   * @param io
   */
  public async listen(io: socketio.Server): Promise<void> {
    io.on(EventName.CONNECTION, this.registerNewSocket.bind(this))
  }

  /**
   * Registers new socket's listeners
   *
   * @param socket
   */
  public registerNewSocket(socket: SocketServer): void {
    socket.on(EventName.REGISTER, async (data: RegisterMessageType) => {
      try {
        await this.register(data, socket)
      } catch (error) {
        Logger.error(error)
      }
    })

    socket.on(EventName.ACTION_SEND, async (data: SyncActionMessageType) => {
      try {
        await this.actionUpdate(data, socket)
      } catch (error) {
        Logger.error(error)
      }
    })

    socket.on(EventName.DISCONNECT, async () => {
      await this.userManager.deregisterUser(socket.id)
    })
  }

  /**
   * Load the correct state and subscribe to redis
   *
   * @param data
   * @param socket
   */
  public async register(
    data: RegisterMessageType,
    socket: SocketServer
  ): Promise<void> {
    const projectName = data.projectName
    const taskId = index2str(data.taskIndex)
    const sessionId = initSessionId(data.sessionId)

    await this.userManager.registerUser(socket.id, projectName, data.userId)

    const state = await this.projectStore.loadState(projectName, taskId)
    state.session.id = sessionId
    state.task.config.autosave = this.autosave
    state.task.config.bots = this.bots

    // Connect socket to others in the same room
    const room = path.getRoomName(projectName, taskId, this.sync, sessionId)
    socket.join(room)
    // Send backend state to newly registered socket
    socket.emit(EventName.REGISTER_ACK, state)
    // Notify other processes of registration
    this.publisher.publishRegisterEvent(data)
  }

  /**
   * Updates the state with the action, and broadcasts action
   *
   * @param data
   * @param socket
   */
  public async actionUpdate(
    data: SyncActionMessageType,
    socket: SocketServer
  ): Promise<void> {
    const projectName = data.projectName
    const taskId = data.taskId
    const sessionId = data.sessionId
    const actions = data.actions.actions
    const actionPacketId = data.actions.id
    const triggerId = data.actions.triggerId

    const room = path.getRoomName(projectName, taskId, this.sync, sessionId)

    const taskActions = actions.filter((action) => {
      return actionConsts.isTaskAction(action)
    })
    const actionTypes = Array.from(new Set(actions.map((a) => a.type)).values())

    Logger.debug(
      `Received ${actions.length} actions and ` +
        `${taskActions.length} task actions from Session ` +
        `${sessionId}. The action types are [${actionTypes.join(", ")}].`
    )

    // Load IDs of actions that have been processed already
    const redisMetadata = await this.projectStore.loadStateMetadata(
      projectName,
      taskId
    )
    const actionIdsSaved = redisMetadata.actionIds
    if (!(actionPacketId in actionIdsSaved) && taskActions.length > 0) {
      const state = await this.projectStore.loadState(projectName, taskId)
      const [newState, timestamps] = updateStateTimestamp(state, taskActions)

      // Mark the id as saved, and store the timestamps
      actionIdsSaved[actionPacketId] = timestamps
      const stateMetadata: StateMetadata = {
        projectName,
        taskId,
        actionIds: actionIdsSaved
      }

      await this.projectStore.saveState(
        newState,
        projectName,
        taskId,
        stateMetadata
      )
    } else if (taskActions.length > 0) {
      // If actions were already saved, apply the old timestamps
      const timestamps = actionIdsSaved[actionPacketId]
      for (let actionInd = 0; actionInd < taskActions.length; actionInd++) {
        taskActions[actionInd].timestamp = timestamps[actionInd]
      }
    }

    if (taskActions.length > 0) {
      // Broadcast task actions to all other sessions in room
      const taskActionMsg: SyncActionMessageType = _.cloneDeep(data)
      taskActionMsg.actions = {
        actions: taskActions,
        id: actionPacketId,
        triggerId
      }
      // Broadcast task actions to all other sessions in room
      socket.broadcast.to(room).emit(EventName.ACTION_BROADCAST, taskActionMsg)
    }
    // Echo everything to original session
    socket.emit(EventName.ACTION_BROADCAST, data)
  }
}

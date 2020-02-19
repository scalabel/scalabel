import socketio from 'socket.io'
import uuid4 from 'uuid/v4'
import * as types from '../action/types'
import Logger from './logger'
import * as path from './path'
import { ProjectStore } from './project_store'
import { SocketServer } from './socket_server'
import {
  ActionPacketType, Env, EventName,
  RegisterMessageType, StateMetadata, SyncActionMessageType } from './types'
import { UserManager } from './user_manager'
import { index2str, updateState } from './util'

/**
 * Wraps socket.io handlers for saving, loading, and synchronization
 */
export class Hub {
  /** flag for sync */
  protected sync: boolean
  /** flag for autosave */
  protected autosave: boolean
  /** the project store */
  protected projectStore: ProjectStore
  /** the user manager */
  protected userManager: UserManager

  constructor (env: Env, projectStore: ProjectStore, userManager: UserManager) {
    this.sync = env.sync
    this.autosave = env.autosave
    this.projectStore = projectStore
    this.userManager = userManager
  }

  /**
   * Listens for websocket connections
   */
  public listen (io: socketio.Server) {
    io.on(EventName.CONNECTION, this.registerNewSocket.bind(this))
  }

  /**
   * Registers new socket's listeners
   */
  public registerNewSocket (socket: SocketServer) {
    socket.on(EventName.REGISTER, async (rawData: string) => {
      try {
        await this.register(rawData, socket)
      } catch (error) {
        Logger.error(error)
      }
    })

    socket.on(EventName.ACTION_SEND, async (rawData: string) => {
      try {
        await this.actionUpdate(rawData, socket)
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
   */
  public async register (rawData: string, socket: SocketServer) {
    const data: RegisterMessageType = JSON.parse(rawData)
    const projectName = data.projectName
    const taskIndex = data.taskIndex
    let sessionId = data.sessionId

    const taskId = index2str(taskIndex)
    await this.userManager.registerUser(socket.id, projectName, data.userId)
    // keep session id if it exists, i.e. if it is a reconnection
    if (!sessionId) {
      // new session on new load
      sessionId = uuid4()
    }
    const state = await this.projectStore.loadState(projectName, taskId)
    state.session.id = sessionId
    state.task.config.autosave = this.autosave

    // Connect socket to others in the same room
    const room = path.getRoomName(projectName, taskId, this.sync, sessionId)
    socket.join(room)
    // Send backend state to newly registered socket
    socket.emit(EventName.REGISTER_ACK, state)
  }

  /**
   * Updates the state with the action, and broadcasts action
   */
  public async actionUpdate (rawData: string, socket: SocketServer) {
    const data: SyncActionMessageType = JSON.parse(rawData)
    const projectName = data.projectName
    const taskId = data.taskId
    const sessionId = data.sessionId
    const actions = data.actions.actions
    const actionPacketId = data.actions.id

    const room = path.getRoomName(projectName, taskId, this.sync, sessionId)

    const taskActions = actions.filter((action) => {
      return types.TASK_ACTION_TYPES.includes(action.type)
    })

    // Load IDs of actions that have been processed already
    const redisMetadata =
      await this.projectStore.loadStateMetadata(projectName, taskId)
    const actionIdsSaved = redisMetadata.actionIds
    // TODO: in the else, case, apply the old timestamps to the same actions

    if (!(actionPacketId in actionIdsSaved) && taskActions.length > 0) {
      const state = await this.projectStore.loadState(projectName, taskId)
      const newState = updateState(state, taskActions)

      // convert set to a list in JSON
      actionIdsSaved[actionPacketId] = timestamps
      const stateMetadata: StateMetadata = {
        projectName,
        taskId,
        actionIds: actionIdsSaved
      }
      await this.projectStore.saveState(
        newState, projectName, taskId, stateMetadata)
    }

    if (taskActions.length > 0) {
      // broadcast task actions to all other sessions in room
      const taskActionMsg: ActionPacketType = {
        actions: taskActions,
        id: actionsId
      }
      // broadcast task actions to all other sessions in room
      socket.broadcast.to(room).emit(EventName.ACTION_BROADCAST, taskActionMsg)
    }
    // echo everything to original session
    console.log(data.actions)
    socket.emit(EventName.ACTION_BROADCAST, data.actions)
  }
}

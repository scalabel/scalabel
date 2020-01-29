import * as socketio from 'socket.io'
import * as uuid4 from 'uuid/v4'
import * as types from '../action/types'
import { configureStore } from '../common/configure_store'
import { ItemTypeName, LabelTypeName, TrackPolicyType } from '../common/types'
import { makeItemStatus, makeState } from '../functional/states'
import { State, TaskType } from '../functional/types'
import Logger from './logger'
import * as path from './path'
import { RedisCache } from './redis_cache'
import Session from './server_session'
import { EventName, RegisterMessageType, SyncActionMessageType } from './types'
import { getSavedKey, getTaskKey,
  index2str, loadSavedState} from './util'

/**
 * Starts socket.io handlers for saving, loading, and synchronization
 */
export function startSocketServer (io: socketio.Server) {
  const cache = new RedisCache()

  io.on(EventName.CONNECTION, (socket: socketio.Socket) => {
    socket.on(EventName.REGISTER, async (rawData: string) => {
      try {
        await register(rawData, socket, cache)
      } catch (error) {
        Logger.error(error)
      }
    })

    socket.on(EventName.ACTION_SEND, async (rawData: string) => {
      try {
        await actionUpdate(rawData, socket, cache)
      } catch (error) {
        Logger.error(error)
      }
    })
  })
}

/**
 * Load the correct state and subscribe to redis
 */
async function register (
  rawData: string, socket: socketio.Socket, cache: RedisCache) {
  const data: RegisterMessageType = JSON.parse(rawData)
  const projectName = data.projectName
  const taskIndex = data.taskIndex
  let sessionId = data.sessionId

  const env = Session.getEnv()
  const taskId = index2str(taskIndex)
  // keep session id if it exists, i.e. if it is a reconnection
  if (!sessionId) {
    // new session on new load
    sessionId = uuid4()
  }
  const state = await loadState(projectName, taskId, cache)
  state.session.id = sessionId
  state.task.config.autosave = env.autosave

  // Connect socket to others in the same room
  const room = path.roomName(projectName, taskId, env.sync, sessionId)
  socket.join(room)
  // Send backend state to newly registered socket
  socket.emit(EventName.REGISTER_ACK, state)
}

/**
 * Updates the state with the action, and broadcasts action
 */
async function actionUpdate (
  rawData: string, socket: socketio.Socket, cache: RedisCache) {
  const data: SyncActionMessageType = JSON.parse(rawData)
  const projectName = data.projectName
  const taskId = data.taskId
  const sessionId = data.sessionId
  const actionList = data.actions
  const env = Session.getEnv()

  const room = path.roomName(projectName, taskId, env.sync, sessionId)

  const taskActions = actionList.filter((action) => {
    return types.TASK_ACTION_TYPES.includes(action.type)
  })

  const state = await loadState(projectName, taskId, cache)
  const store = configureStore(state)

  // For each task action, update the backend store
  for (const action of taskActions) {
    action.timestamp = Date.now()
    store.dispatch(action)
  }

  // broadcast task actions to all other sessions in room
  socket.broadcast.to(room).emit(EventName.ACTION_BROADCAST, taskActions)
  // echo everything to original session
  socket.emit(EventName.ACTION_BROADCAST, actionList)

  const newState = store.getState().present
  const stringState = JSON.stringify(newState)
  const saveKey = getSavedKey(projectName, taskId)

  await cache.setExWithReminder(saveKey, stringState)
}

/**
 * Loads state from cache if available, else memory
 */
export async function loadState (
  projectName: string, taskId: string, cache: RedisCache): Promise<State> {
  let state: State

  // first try to load from redis cache
  const saveKey = getSavedKey(projectName, taskId)
  const redisValue = await cache.get(saveKey)
  if (redisValue) {
    state = JSON.parse(redisValue)
  } else {
    // otherwise load from storage
    try {
      // first, attempt loading previous submission
      state = await loadSavedState(projectName, taskId)
    } catch {
      // if no submissions exist, load from task
      state = await loadStateFromTask(projectName, taskId)
    }
  }
  return state
}

/**
 * Loads the state from task.json (created at import)
 * Used for first load
 */
async function loadStateFromTask (
  projectName: string, taskId: string): Promise<State> {
  const key = getTaskKey(projectName, taskId)
  const fields = await Session.getStorage().load(key)
  const task = JSON.parse(fields) as TaskType
  const state = makeState({ task })

  state.session.itemStatuses = []
  for (const item of state.task.items) {
    const itemStatus = makeItemStatus()
    for (const sensorKey of Object.keys(item.urls)) {
      itemStatus.sensorDataLoaded[Number(sensorKey)] = false
    }
    state.session.itemStatuses.push(itemStatus)
  }

  // TODO: Move this to be in front end after implementing label selector
  switch (state.task.config.itemType) {
    case ItemTypeName.IMAGE:
    case ItemTypeName.VIDEO:
      if (state.task.config.labelTypes.length === 1) {
        switch (state.task.config.labelTypes[0]) {
          case LabelTypeName.BOX_2D:
            state.task.config.policyTypes =
              [TrackPolicyType.LINEAR_INTERPOLATION]
            break
          case LabelTypeName.POLYGON_2D:
            state.task.config.policyTypes =
              [TrackPolicyType.LINEAR_INTERPOLATION]
            break
          case LabelTypeName.CUSTOM_2D:
            state.task.config.labelTypes[0] =
              Object.keys(state.task.config.label2DTemplates)[0]
            state.task.config.policyTypes =
              [TrackPolicyType.LINEAR_INTERPOLATION]
        }
      }
      break
    case ItemTypeName.POINT_CLOUD:
    case ItemTypeName.POINT_CLOUD_TRACKING:
      if (state.task.config.labelTypes.length === 1 &&
          state.task.config.labelTypes[0] === LabelTypeName.BOX_3D) {
        state.task.config.policyTypes =
          [TrackPolicyType.LINEAR_INTERPOLATION]
      }
      break
    case ItemTypeName.FUSION:
      state.task.config.labelTypes =
        [LabelTypeName.BOX_3D, LabelTypeName.PLANE_3D]
      state.task.config.policyTypes = [TrackPolicyType.LINEAR_INTERPOLATION]
      break
  }
  return state
}

import { Store } from 'redux'
import * as socketio from 'socket.io'
import * as uuid4 from 'uuid/v4'
import * as types from '../action/types'
import { configureStore } from '../common/configure_store'
import { ItemTypeName, LabelTypeName, TrackPolicyType } from '../common/types'
import { makeItemStatus, makeState } from '../functional/states'
import { State, TaskType } from '../functional/types'
import * as path from './path'
import Session from './server_session'
import { EventName, RegisterMessageType, SyncActionMessageType } from './types'
import { getSavedKey, getTaskKey,
  index2str, loadSavedState} from './util'

/**
 * Starts socket.io handlers for saving, loading, and synchronization
 */
export function startSocketServer (io: socketio.Server) {
  const env = Session.getEnv()
  // maintain a store for each task
  const stores: { [key: string]: Store } = {}
  io.on(EventName.CONNECTION, (socket: socketio.Socket) => {
    socket.on(EventName.REGISTER, async (rawData: string) => {
      const data: RegisterMessageType = JSON.parse(rawData)
      const projectName = data.projectName

      const taskIndex = data.taskIndex
      const taskId = index2str(taskIndex)

      let sessionId = data.sessionId
      // keep session id if it exists, i.e. if it is a reconnection
      if (!sessionId) {
        // new session on new load
        sessionId = uuid4()
      }

      let room = path.roomName(projectName, taskId, env.sync)

      let state: State

      // if sync is on, try to load from memory
      if (env.sync && room in stores) {
        // try load from memory
        state = stores[room].getState().present
      } else {
        // load from storage
        try {
          // first, attempt loading previous submission
          state = await loadSavedState(projectName, taskId)
        } catch {
          // if no submissions exist, load from task
          state = await loadStateFromTask(projectName, taskIndex)
        }

        // update room with loaded session Id
        room = path.roomName(projectName, taskId,
          env.sync, sessionId)

        // update memory with new state
        stores[room] = configureStore(state)
      }
      state.session.id = sessionId
      state.task.config.autosave = env.autosave

      // Connect socket to others in the same room
      socket.join(room)
      // Send backend state to newly registered socket
      socket.emit(EventName.REGISTER_ACK, state)
    })

    socket.on(EventName.ACTION_SEND, async (rawData: string) => {
      const data: SyncActionMessageType = JSON.parse(rawData)
      const projectName = data.projectName
      const taskId = data.taskId
      const sessionId = data.sessionId
      const actionList = data.actions

      const room = path.roomName(projectName, taskId, env.sync, sessionId)

      const taskActions = actionList.filter((action) => {
        return types.TASK_ACTION_TYPES.includes(action.type)
      })

      // For each task action, update the backend store
      for (const action of taskActions) {
        action.timestamp = Date.now()
        stores[room].dispatch(action)
      }
      // broadcast task actions to all other sessions in room
      socket.broadcast.to(room).emit(EventName.ACTION_BROADCAST, taskActions)
      // echo everything to original session
      socket.emit(EventName.ACTION_BROADCAST, actionList)

      // save task data with all updates
      const content = JSON.stringify(stores[room].getState().present)
      const filePath = path.getFileKey(getSavedKey(projectName, taskId))
      await Session.getStorage().save(filePath, content)
    })
  })
}

/**
 * Loads the state from task.json (created at import)
 * Used for first load
 */
async function loadStateFromTask (
  projectName: string,
  taskIndex: number): Promise<State> {
  const key = getTaskKey(projectName, index2str(taskIndex))
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

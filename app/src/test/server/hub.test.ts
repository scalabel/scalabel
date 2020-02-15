import { cleanup } from '@testing-library/react'
import { goToItem } from '../../js/action/common'
import { FileStorage } from '../../js/server/file_storage'
import { Hub } from '../../js/server/hub'
import { ProjectStore } from '../../js/server/project_store'
import { SocketServer } from '../../js/server/socket_server'
import { ActionPacketType, defaultEnv, EventName,
  RegisterMessageType, SyncActionMessageType } from '../../js/server/types'
import { UserManager } from '../../js/server/user_manager'
import { index2str, updateState } from '../../js/server/util'
import { getInitialState, getRandomBox2dAction } from '../util'

jest.mock('../../js/server/file_storage')
jest.mock('../../js/server/path')
jest.mock('../../js/server/project_store')
jest.mock('../../js/server/user_manager')

let socketId: string
let projectName: string
let taskIndex: number
let taskId: string
let sessionId: string
let userId: string
let actionListId: string
let mockSocket: SocketServer
let mockStorage: FileStorage
let mockProjectStore: ProjectStore
let mockUserManager: UserManager
let hub: Hub
let broadcastFunc: (event: string, data: ActionPacketType) => void

beforeAll(() => {
  const constantDate = Date.now()
  Date.now = jest.fn(() => {
    return constantDate
  })

  socketId = 'socketId'
  projectName = 'testProject'
  taskIndex = 0
  taskId = index2str(taskIndex)
  sessionId = 'testSessId'
  userId = 'testUserId'
  actionListId = 'actionListId'
  broadcastFunc = jest.fn()

  mockSocket = {
    on: jest.fn(),
    emit: jest.fn(),
    join: jest.fn(),
    broadcast: {
      to: jest.fn().mockImplementation(() => {
        return {
          emit: broadcastFunc
        }
      })
    },
    id: socketId
  }
  mockStorage = new FileStorage('fakeDataDir')
  mockProjectStore = new ProjectStore(mockStorage)
  mockUserManager = new UserManager(mockStorage)
  hub = new Hub(defaultEnv, mockProjectStore, mockUserManager)
})

afterEach(cleanup)
describe('Test hub functionality', () => {
  beforeEach(() => {
    jest.clearAllMocks()

    mockProjectStore.loadState = jest.fn().mockImplementation(() => {
      return getInitialState(sessionId)
    })

    const initialMetadata = {
      projectName,
      taskId,
      actionIds: []
    }
    mockProjectStore.loadStateMetadata = jest.fn().mockImplementation(() => {
      return initialMetadata
    })
  })

  test('Test registration', async () => {
    const data: RegisterMessageType = {
      projectName,
      taskIndex,
      sessionId,
      userId
    }
    const rawData = JSON.stringify(data)
    await hub.register(rawData, mockSocket)
    expect(mockUserManager.registerUser).toBeCalledWith(
      socketId, projectName, userId)
    expect(mockSocket.join).toBeCalled()
    expect(mockSocket.emit).toBeCalledWith(
      EventName.REGISTER_ACK, getInitialState(sessionId)
    )
  })

  test('Test task action update saves data and broadcasts', async () => {
    // make a task action
    const action = getRandomBox2dAction()
    const data: SyncActionMessageType = {
      projectName,
      taskId,
      sessionId,
      actions: {
        actions: [action],
        id: actionListId
      }
    }
    const newMetadata = {
      projectName,
      taskId,
      actionIds: [actionListId]
    }

    const rawData = JSON.stringify(data)
    await hub.actionUpdate(rawData, mockSocket)
    const newState = updateState(getInitialState(sessionId), [action])
    expect(mockProjectStore.saveState).toBeCalledWith(newState, projectName,
      taskId, newMetadata)
    expect(broadcastFunc).toBeCalledWith(
      EventName.ACTION_BROADCAST, data.actions)
    expect(mockSocket.emit).toBeCalledWith(
      EventName.ACTION_BROADCAST, data.actions)
  })

  test('Non-task action just echoes', async () => {
    // make a non-task action
    const action = goToItem(0)
    const data: SyncActionMessageType = {
      projectName,
      taskId,
      sessionId,
      actions: {
        actions: [action],
        id: actionListId
      }
    }
    const rawData = JSON.stringify(data)
    await hub.actionUpdate(rawData, mockSocket)
    expect(mockProjectStore.saveState).not.toBeCalled()
    expect(broadcastFunc).not.toBeCalled()
    expect(mockSocket.emit).toBeCalledWith(
      EventName.ACTION_BROADCAST, data.actions)
  })

  test.only('If saved, repeated message does not save again', async () => {
    // make a task action
    const action = getRandomBox2dAction()
    const data: SyncActionMessageType = {
      projectName,
      taskId,
      sessionId,
      actions: {
        actions: [action],
        id: actionListId
      }
    }
    const newMetadata = {
      projectName,
      taskId,
      actionIds: [actionListId]
    }

    const rawData = JSON.stringify(data)
    await hub.actionUpdate(rawData, mockSocket)
    expect(mockProjectStore.saveState).toHaveBeenCalledTimes(1)
    expect(broadcastFunc).toHaveBeenCalledTimes(1)
    expect(mockSocket.emit).toHaveBeenCalledTimes(1)

    const newState = updateState(getInitialState(sessionId), [action])
    mockProjectStore.loadState = jest.fn().mockImplementation(() => {
      return newState
    })
    mockProjectStore.loadStateMetadata = jest.fn().mockImplementation(() => {
      return newMetadata
    })
    await hub.actionUpdate(rawData, mockSocket)

    expect(mockProjectStore.saveState).toHaveBeenCalledTimes(1)
    expect(broadcastFunc).toHaveBeenCalledTimes(2)
    expect(mockSocket.emit).toHaveBeenCalledTimes(2)

    // TODO: verify that both calls to emit use actions with the same timestamps
  })

  test('If crash before saving, saves again', async () => {
    // make a task action
    const action = getRandomBox2dAction()
    const data: SyncActionMessageType = {
      projectName,
      taskId,
      sessionId,
      actions: {
        actions: [action],
        id: actionListId
      }
    }

    const rawData = JSON.stringify(data)
    await hub.actionUpdate(rawData, mockSocket)
    expect(mockProjectStore.saveState).toHaveBeenCalledTimes(1)
    expect(broadcastFunc).toHaveBeenCalledTimes(1)
    expect(mockSocket.emit).toHaveBeenCalledTimes(1)

    // model crash by not updating state or metadata that gets loaded
    await hub.actionUpdate(rawData, mockSocket)

    expect(mockProjectStore.saveState).toHaveBeenCalledTimes(2)
    expect(broadcastFunc).toHaveBeenCalledTimes(2)
    expect(mockSocket.emit).toHaveBeenCalledTimes(2)
  })
})

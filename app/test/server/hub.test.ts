import { cleanup } from "@testing-library/react"
import _ from "lodash"

import { goToItem } from "../../src/action/common"
import { index2str } from "../../src/common/util"
import { EventName } from "../../src/const/connection"
import { serverConfig } from "../../src/server/defaults"
import { FileStorage } from "../../src/server/file_storage"
import { Hub } from "../../src/server/hub"
import { ProjectStore } from "../../src/server/project_store"
import { RedisCache } from "../../src/server/redis_cache"
import { RedisClient } from "../../src/server/redis_client"
import { RedisPubSub } from "../../src/server/redis_pub_sub"
import { UserManager } from "../../src/server/user_manager"
import { updateState } from "../../src/server/util"
import {
  ActionPacketType,
  RegisterMessageType,
  SyncActionMessageType
} from "../../src/types/message"
import { StateMetadata } from "../../src/types/project"
import { getInitialState, getRandomBox2dAction } from "./util/util"

jest.mock("../../src/server/file_storage")
jest.mock("../../src/server/path")
jest.mock("../../src/server/project_store")
jest.mock("../../src/server/user_manager")
jest.mock("../../src/server/redis_client")
jest.mock("../../src/server/redis_pub_sub")

let projectName: string
let taskIndex: number
let taskId: string
let sessionId: string
let userId: string
let actionListId: string
let mockStorage: FileStorage
let mockProjectStore: ProjectStore
let mockUserManager: UserManager
let mockPubSub: RedisPubSub
let hub: Hub
const broadcastFunc = jest.fn()
const socketId = "socketId"
const mockSocket = {
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

beforeAll(() => {
  projectName = "testProject"
  taskIndex = 0
  taskId = index2str(taskIndex)
  sessionId = "testSessionId"
  userId = "testUserId"
  actionListId = "actionListId"

  mockStorage = new FileStorage("fakeDataDir")
  const client = new RedisClient(serverConfig.redis)
  const redisStore = new RedisCache(serverConfig.redis, mockStorage, client)
  mockPubSub = new RedisPubSub(client)
  mockProjectStore = new ProjectStore(mockStorage, redisStore)
  mockUserManager = new UserManager(mockProjectStore)
  hub = new Hub(serverConfig, mockProjectStore, mockUserManager, mockPubSub)
})

afterEach(cleanup)

beforeEach(() => {
  jest.clearAllMocks()

  mockProjectStore.loadState = jest.fn().mockImplementation(() => {
    return getInitialState(sessionId)
  })

  const initialMetadata = {
    projectName,
    taskId,
    actionIds: {}
  }
  mockProjectStore.loadStateMetadata = jest.fn().mockImplementation(() => {
    return _.cloneDeep(initialMetadata)
  })
})

describe("Test hub functionality", () => {
  test("Test registration", async () => {
    const data: RegisterMessageType = {
      projectName,
      taskIndex,
      sessionId,
      userId,
      address: "",
      bot: false
    }
    await hub.register(data, mockSocket)
    expect(mockUserManager.registerUser).toBeCalledWith(
      socketId,
      projectName,
      userId
    )
    expect(mockPubSub.publishRegisterEvent).toBeCalledWith(data)
    expect(mockSocket.join).toBeCalled()
    expect(mockSocket.emit).toBeCalledWith(
      EventName.REGISTER_ACK,
      getInitialState(sessionId)
    )
  })

  test("Test task action update saves data and broadcasts", async () => {
    // Mock date for action timestamp
    const constantDate = Date.now()
    const dateFn = Date.now
    Date.now = jest.fn(() => {
      return constantDate
    })

    // Make a task action
    const action = getRandomBox2dAction()
    const data: SyncActionMessageType = {
      projectName,
      taskId,
      sessionId,
      actions: {
        actions: [action],
        id: actionListId
      },
      bot: false
    }

    // Send the action
    await hub.actionUpdate(data, mockSocket)

    // Test that state/metadata updated correctly
    const newMetadata = {
      projectName,
      taskId,
      actionIds: {
        actionListId: [constantDate]
      }
    }
    const newState = updateState(getInitialState(sessionId), [action])
    expect(mockProjectStore.saveState).toBeCalledWith(
      newState,
      projectName,
      taskId,
      newMetadata
    )

    // Test that actions were broadcast correctly
    const newAction = _.cloneDeep(action)
    newAction.timestamp = constantDate
    const newPacket: ActionPacketType = {
      actions: [newAction],
      id: actionListId
    }
    expect(broadcastFunc).toBeCalledWith(
      EventName.ACTION_BROADCAST,
      packetToMessage(newPacket)
    )
    expect(mockSocket.emit).toBeCalledWith(
      EventName.ACTION_BROADCAST,
      packetToMessage(newPacket)
    )

    // Restore the date function
    Date.now = dateFn
  })

  test("Non-task action just echoes", async () => {
    // Make a non-task action
    const action = goToItem(0)
    const data: SyncActionMessageType = {
      projectName,
      taskId,
      sessionId,
      actions: {
        actions: [action],
        id: actionListId
      },
      bot: false
    }
    await hub.actionUpdate(data, mockSocket)
    expect(mockProjectStore.saveState).not.toBeCalled()
    expect(broadcastFunc).not.toBeCalled()
    expect(mockSocket.emit).toBeCalledWith(EventName.ACTION_BROADCAST, data)
  })

  test("If saved, repeated message does not save again", async () => {
    // Make a task action
    const action = getRandomBox2dAction()
    const data: SyncActionMessageType = {
      projectName,
      taskId,
      sessionId,
      actions: {
        actions: [action],
        id: actionListId
      },
      bot: false
    }

    // Send message for the first time
    await hub.actionUpdate(data, mockSocket)

    expect(mockProjectStore.saveState).toHaveBeenCalledTimes(1)
    expect(broadcastFunc).toHaveBeenCalledTimes(1)
    expect(mockSocket.emit).toHaveBeenCalledTimes(1)
    const packet: ActionPacketType = mockSocket.emit.mock.calls[0][1].actions
    const timestamp = packet.actions[0].timestamp

    // Send message for the second time, using updates values
    const newState = updateState(getInitialState(sessionId), [action])
    const newMetadata: StateMetadata = {
      projectName,
      taskId,
      actionIds: {
        actionListId: [timestamp]
      }
    }
    mockProjectStore.loadState = jest.fn().mockImplementation(() => {
      return newState
    })
    mockProjectStore.loadStateMetadata = jest.fn().mockImplementation(() => {
      return newMetadata
    })

    await hub.actionUpdate(data, mockSocket)

    expect(mockProjectStore.saveState).toHaveBeenCalledTimes(1)
    expect(broadcastFunc).toHaveBeenCalledTimes(2)
    expect(mockSocket.emit).toHaveBeenCalledTimes(2)

    // Verify that the actions have the same timestamps in both emit calls
    const calls = mockSocket.emit.mock.calls
    expect(calls[0]).toStrictEqual(calls[1])
  })

  test("If crash before saving, saves again", async () => {
    // Make a task action
    const action = getRandomBox2dAction()
    const data: SyncActionMessageType = {
      projectName,
      taskId,
      sessionId,
      actions: {
        actions: [action],
        id: actionListId
      },
      bot: false
    }

    await hub.actionUpdate(data, mockSocket)
    expect(mockProjectStore.saveState).toHaveBeenCalledTimes(1)
    expect(broadcastFunc).toHaveBeenCalledTimes(1)
    expect(mockSocket.emit).toHaveBeenCalledTimes(1)

    // Model crash by not updating state or metadata that gets loaded
    await hub.actionUpdate(data, mockSocket)

    expect(mockProjectStore.saveState).toHaveBeenCalledTimes(2)
    expect(broadcastFunc).toHaveBeenCalledTimes(2)
    expect(mockSocket.emit).toHaveBeenCalledTimes(2)
  })
})

/**
 * Convert action packet to sync message
 *
 * @param packet
 */
function packetToMessage(packet: ActionPacketType): SyncActionMessageType {
  return {
    actions: packet,
    projectName,
    sessionId,
    taskId: index2str(taskIndex),
    bot: false
  }
}

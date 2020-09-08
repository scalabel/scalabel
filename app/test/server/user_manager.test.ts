import * as fs from "fs-extra"
import _ from "lodash"
import mockfs from "mock-fs"

import { uid } from "../../src/common/uid"
import { serverConfig } from "../../src/server/defaults"
import { FileStorage } from "../../src/server/file_storage"
import { getTestDir } from "../../src/server/path"
import { ProjectStore } from "../../src/server/project_store"
import { RedisCache } from "../../src/server/redis_cache"
import { RedisClient } from "../../src/server/redis_client"
import { UserManager } from "../../src/server/user_manager"
import { makeProjectDir } from "./util/util"

jest.mock("../../src/server/redis_client")

let userManager: UserManager
let projectName: string
let projectName2: string
let dataDir: string

beforeAll(() => {
  projectName = "myProject"
  projectName2 = "myProject2"
  dataDir = getTestDir("test-user-data")
  makeProjectDir(dataDir, projectName)
  const storage = new FileStorage(dataDir)
  const client = new RedisClient(serverConfig.redis)
  const redisStore = new RedisCache(serverConfig.redis, storage, client)
  const projectStore = new ProjectStore(storage, redisStore)
  userManager = new UserManager(projectStore)
})

afterAll(() => {
  fs.removeSync(dataDir)
})

describe("test user management", () => {
  test("user count works for single user", async () => {
    // Note: this is not actually how user ids are generated
    const userId = uid()
    const socketId = uid()
    expect(await userManager.countUsers(projectName)).toBe(0)
    await userManager.registerUser(socketId, projectName, userId)
    expect(await userManager.countUsers(projectName)).toBe(1)
    await userManager.deregisterUser(socketId)
    expect(await userManager.countUsers(projectName)).toBe(0)
  })

  test("user count works for multiple users", async () => {
    const numUsers = 3
    const socketsPerUser = 3
    const userIds = _.range(numUsers).map(() => uid())
    // First 3 socket ids correspond to first user, etc.
    const socketIds = _.range(numUsers * socketsPerUser).map(() => uid())
    for (let socketNum = 0; socketNum < socketsPerUser; socketNum++) {
      for (let userNum = 0; userNum < numUsers; userNum++) {
        const socketId = socketIds[userNum * socketsPerUser + socketNum]
        const userId = userIds[userNum]
        await userManager.registerUser(socketId, projectName, userId)
      }
      // Each iteration adds more sockets, but number of users is constant
      expect(await userManager.countUsers(projectName)).toBe(numUsers)
    }

    // Remove all sockets for each user consecutively
    for (let userNum = 0; userNum < numUsers; userNum++) {
      for (let socketNum = 0; socketNum < socketsPerUser; socketNum++) {
        const socketId = socketIds[userNum * socketsPerUser + socketNum]
        await userManager.deregisterUser(socketId)
      }
      const numUsersActual = await userManager.countUsers(projectName)
      expect(numUsersActual).toBe(numUsers - 1 - userNum)
    }
  })

  test("user count works for multiple projects", async () => {
    // Make sure other tests clean up worked
    expect(await userManager.countUsers(projectName)).toBe(0)
    expect(await userManager.countUsers(projectName2)).toBe(0)

    const numUsers = 2
    const numProjects = 2
    const userIds = _.range(numUsers).map(() => uid())
    // First 2 socket ids correspond to first project, etc.
    const socketIds = _.range(numUsers * numProjects).map(() => uid())

    // First put all users on 1st project
    for (let userNum = 0; userNum < numUsers; userNum++) {
      await userManager.registerUser(
        socketIds[userNum],
        projectName,
        userIds[userNum]
      )
    }
    expect(await userManager.countUsers(projectName)).toBe(numUsers)
    expect(await userManager.countUsers(projectName2)).toBe(0)

    // Then move them to second project
    for (let userNum = 0; userNum < numUsers; userNum++) {
      await userManager.deregisterUser(socketIds[userNum])
      await userManager.registerUser(
        socketIds[numUsers + userNum],
        projectName2,
        userIds[userNum]
      )
    }
    expect(await userManager.countUsers(projectName)).toBe(0)
    expect(await userManager.countUsers(projectName2)).toBe(numUsers)

    // Then cleanup
    for (let userNum = 0; userNum < numUsers; userNum++) {
      await userManager.deregisterUser(socketIds[numUsers + userNum])
    }
    expect(await userManager.countUsers(projectName)).toBe(0)
    expect(await userManager.countUsers(projectName2)).toBe(0)
  })

  test("clearing users resets all counts", async () => {
    // Make sure other tests clean up worked
    expect(await userManager.countUsers(projectName)).toBe(0)
    expect(await userManager.countUsers(projectName2)).toBe(0)

    const numUsers = 2
    const numProjects = 2
    const userIds = _.range(numUsers).map(() => uid())
    // First 2 socket ids correspond to first project, etc.
    const socketIds = _.range(numUsers * numProjects).map(() => uid())

    // Put 1 user on each project
    await userManager.registerUser(socketIds[0], projectName, userIds[0])
    await userManager.registerUser(socketIds[1], projectName2, userIds[1])
    expect(await userManager.countUsers(projectName)).toBe(1)
    expect(await userManager.countUsers(projectName2)).toBe(1)

    // Then test that clearing works
    await userManager.clearUsers()
    expect(await userManager.countUsers(projectName)).toBe(0)
    expect(await userManager.countUsers(projectName2)).toBe(0)
  })
})

afterAll(() => {
  mockfs.restore()
})

import * as fs from 'fs-extra'
import _ from 'lodash'
import mockfs from 'mock-fs'
import uuid4 from 'uuid/v4'
import { FileStorage } from '../../js/server/file_storage'
import { getTestDir } from '../../js/server/path'
import { UserManager } from '../../js/server/user_manager'
import { makeProjectDir } from '../util'

let userManager: UserManager
let projectName: string
let dataDir: string

beforeAll(() => {
  projectName = 'myProject'
  dataDir = getTestDir('test-user-data')
  makeProjectDir(dataDir, projectName)
  const storage = new FileStorage(dataDir)
  userManager = new UserManager(storage)
})

afterAll(() => {
  fs.removeSync(dataDir)
})

describe('test user management', () => {

  test('user count works for single user', async () => {
    // note: this is not actually how user id is generated
    const userId = uuid4()
    const socketId = uuid4()
    expect(await userManager.countUsers(projectName)).toBe(0)
    await userManager.registerUser(socketId, projectName, userId)
    expect(await userManager.countUsers(projectName)).toBe(1)
    await userManager.deregisterUser(socketId)
    expect(await userManager.countUsers(projectName)).toBe(0)
  })

  test('user count works for multiple users', async () => {
    const numUsers = 3
    const socketsPerUser = 3
    const userIds = _.range(numUsers).map(() => uuid4())
    // first 3 socket ids correspond to first user, etc.
    const socketIds = _.range(numUsers * socketsPerUser).map(() => uuid4())
    for (let socketNum = 0; socketNum < socketsPerUser; socketNum++) {
      for (let userNum = 0; userNum < numUsers; userNum++) {
        const socketId = socketIds[userNum * socketsPerUser + socketNum]
        const userId = userIds[userNum]
        await userManager.registerUser(socketId, projectName, userId)
      }
      // each iteration adds more sockets, but number of users is constant
      expect(await userManager.countUsers(projectName)).toBe(numUsers)
    }

    // remove all sockets for each user consecutively
    for (let userNum = 0; userNum < numUsers; userNum++) {
      for (let socketNum = 0; socketNum < socketsPerUser; socketNum++) {
        const socketId = socketIds[userNum * socketsPerUser + socketNum]
        await userManager.deregisterUser(socketId)
      }
      const numUsersActual = await userManager.countUsers(projectName)
      expect(numUsersActual).toBe(numUsers - 1 - userNum)
    }
  })

  test('user count works for multiple projects', async () => {
    const projectName2 = 'testProject2'

    // make sure other tests clean up worked
    expect(await userManager.countUsers(projectName)).toBe(0)
    expect(await userManager.countUsers(projectName2)).toBe(0)

    const numUsers = 2
    const numProjects = 2
    const userIds = _.range(numUsers).map(() => uuid4())
    // first 2 socket ids correspond to first project, etc.
    const socketIds = _.range(numUsers * numProjects).map(() => uuid4())

    // First put all users on 1st project
    for (let userNum = 0; userNum < numUsers; userNum++) {
      await userManager.registerUser(
        socketIds[userNum], projectName, userIds[userNum])
    }
    expect(await userManager.countUsers(projectName)).toBe(numUsers)
    expect(await userManager.countUsers(projectName2)).toBe(0)

    // Then move them to second project
    for (let userNum = 0; userNum < numUsers; userNum++) {
      await userManager.deregisterUser(socketIds[userNum])
      await userManager.registerUser(socketIds[numUsers + userNum],
        projectName2, userIds[userNum])
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
})

afterAll(() => {
  mockfs.restore()
})

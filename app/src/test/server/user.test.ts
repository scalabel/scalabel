import _ from 'lodash'
import mockfs from 'mock-fs'
import uuid4 from 'uuid/v4'
import { countUsers, deregisterUser, registerUser } from '../../js/server/user'

beforeAll(() => {
  // mock the file system for testing storage
  mockfs({
    'data/myProject': {
      '.config': 'config contents',
      'project.json': 'project contents',
      'tasks': {
        '000000.json': '{"testField": "testValue"}',
        '000001.json': 'contents 1'
      }
    }
  })
})

describe('test user management', () => {
  const projectName = 'myProject'

  test('user count works for single user', async () => {
    // note: this is not actually how user id is generated
    const userId = uuid4()
    const socketId = uuid4()
    expect(await countUsers(projectName)).toBe(0)
    await registerUser(socketId, projectName, userId)
    expect(await countUsers(projectName)).toBe(1)
    await deregisterUser(socketId)
    expect(await countUsers(projectName)).toBe(0)
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
        await registerUser(socketId, projectName, userId)
      }
      // each iteration adds more sockets, but number of users is constant
      expect(await countUsers(projectName)).toBe(numUsers)
    }

    // remove all sockets for each user consecutively
    for (let userNum = 0; userNum < numUsers; userNum++) {
      for (let socketNum = 0; socketNum < socketsPerUser; socketNum++) {
        await deregisterUser(socketIds[userNum * socketsPerUser + socketNum])
      }
      expect(await countUsers(projectName)).toBe(numUsers - 1 - userNum)
    }
  })

  test('user count works for multiple projects', async () => {
    const projectName2 = 'testProject2'

    // make sure other tests clean up worked
    expect(await countUsers(projectName)).toBe(0)
    expect(await countUsers(projectName2)).toBe(0)

    const numUsers = 2
    const numProjects = 2
    const userIds = _.range(numUsers).map(() => uuid4())
    // first 2 socket ids correspond to first project, etc.
    const socketIds = _.range(numUsers * numProjects).map(() => uuid4())

    // First put all users on 1st project
    for (let userNum = 0; userNum < numUsers; userNum++) {
      await registerUser(socketIds[userNum], projectName, userIds[userNum])
    }
    expect(await countUsers(projectName)).toBe(numUsers)
    expect(await countUsers(projectName2)).toBe(0)

    // Then move them to second project
    for (let userNum = 0; userNum < numUsers; userNum++) {
      await deregisterUser(socketIds[userNum])
      await registerUser(socketIds[numUsers + userNum],
        projectName2, userIds[userNum])
    }
    expect(await countUsers(projectName)).toBe(0)
    expect(await countUsers(projectName2)).toBe(numUsers)

    // Then cleanup
    for (let userNum = 0; userNum < numUsers; userNum++) {
      await deregisterUser(socketIds[numUsers + userNum])
    }
    expect(await countUsers(projectName)).toBe(0)
    expect(await countUsers(projectName2)).toBe(0)
  })
})

afterAll(() => {
  mockfs.restore()
})

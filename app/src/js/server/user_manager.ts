import { getMetaKey, getUserKey } from './path'
import { Storage } from './storage'

/**
 * Wraps interface with storage for user management
 */
export class UserManager {
  /** the permanent storage */
  protected storage: Storage

  constructor (storage: Storage) {
    this.storage = storage
  }

  /**
   * Saves the current socket's user data
   */
  public async registerUser (
    socketId: string, projectName: string, userId: string) {
    let socketToUser: { [key: string]: string } = {}
    let userToSockets: { [key: string]: string[] } = {}
    const key = getUserKey(projectName)
    if (await this.storage.hasKey(key)) {
      [socketToUser, userToSockets] = JSON.parse(
        await this.storage.load(key))
    }

    // Update user data with new socket
    let userSockets: string[] = []
    if (userId in userToSockets) {
      userSockets = userToSockets[userId]
    }
    userSockets.push(socketId)
    userToSockets[userId] = userSockets
    socketToUser[socketId] = userId

    const writeData = JSON.stringify([socketToUser, userToSockets])
    await this.storage.save(key, writeData)

    // Update user metadata
    const metaKey = getMetaKey()
    let socketToProject: { [key: string]: string } = {}
    if (await this.storage.hasKey(metaKey)) {
      socketToProject = JSON.parse(await this.storage.load(metaKey))
    }
    socketToProject[socketId] = projectName
    await this.storage.save(metaKey, JSON.stringify(socketToProject))
  }

  /**
   * Deletes the user data of the socket that disconnected
   */
  public async deregisterUser (socketId: string) {
    // First access the projectName via metadata
    const metaKey = getMetaKey()
    if (!(await this.storage.hasKey(metaKey))) {
      return
    }
    const socketToProject = JSON.parse(await this.storage.load(metaKey))
    if (!(socketId in socketToProject)) {
      return
    }
    const projectName = socketToProject[socketId]

    // Next remove the user info for that project
    const key = getUserKey(projectName)
    if (!(await this.storage.hasKey(key))) {
      return
    }
    const [socketToUser, userToSockets]
      = JSON.parse(await this.storage.load(key))
    if (!socketToUser || !(socketId in socketToUser)) {
      return
    }
    const userId = socketToUser[socketId]
    const socketInd = userToSockets[userId].indexOf(socketId)
    if (socketInd > -1) {
      // remove the socket from the user
      userToSockets[userId].splice(socketInd, 1)
    }
    if (userToSockets[userId].length === 0) {
      delete userToSockets[userId]
    }

    delete socketToUser[socketId]

    const writeData = JSON.stringify([socketToUser, userToSockets])
    await this.storage.save(key, writeData)
  }

  /**
   * Counts the number of currently connected users
   */
  public async countUsers (projectName: string): Promise<number> {
    const userKey = getUserKey(projectName)
    let numUsers = 0
    if (await this.storage.hasKey(userKey)) {
      const [, userToSockets] = JSON.parse(
        await this.storage.load(userKey))
      if (userToSockets) {
        numUsers = Object.keys(userToSockets).length
      }
    }
    return numUsers
  }
}

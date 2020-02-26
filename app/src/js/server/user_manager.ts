import _ from 'lodash'
import { getMetaKey, getUserKey } from './path'
import { Storage } from './storage'
import { StringListMap, StringMap, UserData, UserMetadata } from './types'
import { safeParseJSON } from './util'

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
    const key = getUserKey(projectName)
    const userData = await this.loadUserData(key)
    const socketToUser = userData.socketToUser
    const userToSockets = userData.userToSockets

    // Update user with new socket
    let userSockets: string[] = []
    if (userId in userToSockets) {
      userSockets = userToSockets[userId]
    }
    userSockets.push(socketId)
    userToSockets[userId] = userSockets

    // Link socket to user
    socketToUser[socketId] = userId
    await this.saveUser(key, socketToUser, userToSockets)

    // Update user metadata
    const metaKey = getMetaKey()
    const socketToProject = await this.getSocketToProjectMap(metaKey)
    socketToProject[socketId] = projectName
    await this.saveMeta(metaKey, socketToProject)
  }

  /**
   * Deletes the user data of the socket that disconnected
   */
  public async deregisterUser (socketId: string) {
    // Access the projectName via metadata
    const metaKey = getMetaKey()
    const socketToProject = await this.getSocketToProjectMap(metaKey)
    if (!(socketId in socketToProject)) {
      // socket has no associated project
      return
    }
    const projectName = socketToProject[socketId]

    // Remove the socket info from the metadata
    delete socketToProject[socketId]
    await this.saveMeta(metaKey, socketToProject)

    // Next remove the user info for that project
    const key = getUserKey(projectName)
    const userData = await this.loadUserData(key)
    const socketToUser = userData.socketToUser
    const userToSockets = userData.userToSockets

    if (!socketToUser || !(socketId in socketToUser)) {
      // socket has no associated user
      return
    }
    // Remove map from socket to user
    const userId = socketToUser[socketId]
    delete socketToUser[socketId]

    // Remove socket from its user's list
    const socketInd = userToSockets[userId].indexOf(socketId)
    if (socketInd > -1) {
      userToSockets[userId].splice(socketInd, 1)
    }

    // Remove the user if it has no sockets left
    if (userToSockets[userId].length === 0) {
      delete userToSockets[userId]
    }
    await this.saveUser(key, socketToUser, userToSockets)
  }

  /**
   * Counts the number of currently connected users
   */
  public async countUsers (projectName: string): Promise<number> {
    const key = getUserKey(projectName)
    const userData = await this.loadUserData(key)
    const userToSockets = userData.userToSockets
    if (!userToSockets) {
      return 0
    }
    return Object.keys(userToSockets).length
  }

  /**
   * Remove all active users, so all counts should be 0
   */
  public async clearUsers (): Promise<void> {
    // Access the project names from the metadata
    const metaKey = getMetaKey()
    const socketToProject = await this.getSocketToProjectMap(metaKey)
    const activeProjects = Object.values(socketToProject)

    // Clear the metadata
    await this.saveMeta(metaKey, {})

    // Remove the user info for each project
    for (const projectName of activeProjects) {
      const key = getUserKey(projectName)
      await this.saveUser(key, {}, {})
    }
  }

  /**
   * Get map from sockets to projects from metadata
   */
  private async getSocketToProjectMap (metaKey: string): Promise<StringMap> {
    const metaDataJSON = await this.storage.safeLoad(metaKey)
    if (!metaDataJSON) {
      return {}
    }
    // Handle backwards compatability
    const userMetadata = safeParseJSON(metaDataJSON)
    if (_.has(userMetadata, 'socketToProject')) {
      // New code saves as an object, which allows extensions
      return (userMetadata as UserMetadata).socketToProject
    }
    // Old code saved map of projects directly
    return userMetadata as StringMap
  }

  /**
   * Load user data, or return defaults
   */
  private async loadUserData (key: string): Promise<UserData> {
    const userDataJSON = await this.storage.safeLoad(key)
    if (userDataJSON) {
      return safeParseJSON(userDataJSON) as UserData
    }
    return {
      socketToUser: {},
      userToSockets: {}
    }
  }

  /**
   * Save the user data in object format
   */
  private async saveUser (
    key: string, socketToUser: StringMap,
    userToSockets: StringListMap): Promise<void> {
    const userData: UserData = {
      socketToUser,
      userToSockets
    }
    await this.storage.save(key, JSON.stringify(userData))
  }

  /**
   * Save the user metadata in object format
   */
  private async saveMeta (
    metaKey: string, socketToProject: StringMap): Promise<void> {
    const metadata: UserMetadata = {
      socketToProject
    }
    await this.storage.save(metaKey, JSON.stringify(metadata))
  }
}

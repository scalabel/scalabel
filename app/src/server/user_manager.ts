import { UserData, UserMetadata } from "../types/project"
import { ProjectStore } from "./project_store"
import { makeUserData, makeUserMetadata } from "./util"

/**
 * Wraps interface with storage for user management
 */
export class UserManager {
  /** the permanent storage */
  protected projectStore: ProjectStore
  /** whether to apply hotfix that disables user management */
  private readonly disable: boolean

  /**
   * Constructor
   *
   * @param projectStore
   * @param userManagement
   */
  constructor(projectStore: ProjectStore, userManagement: boolean = true) {
    this.projectStore = projectStore
    this.disable = !userManagement
  }

  /**
   * Saves the current socket's user data
   *
   * @param socketId
   * @param projectName
   * @param userId
   */
  public async registerUser(
    socketId: string,
    projectName: string,
    userId: string
  ): Promise<void> {
    if (this.disable) {
      return
    }
    let userData = await this.projectStore.loadUserData(projectName)
    userData = this.addSocketToUser(userData, socketId, userId)
    await this.projectStore.saveUserData(userData)

    let userMetadata = await this.projectStore.loadUserMetadata()
    userMetadata = this.addSocketToMeta(userMetadata, socketId, projectName)
    await this.projectStore.saveUserMetadata(userMetadata)
  }

  /**
   * Deletes the user data of the socket that disconnected
   *
   * @param socketId
   */
  public async deregisterUser(socketId: string): Promise<void> {
    if (this.disable) {
      return
    }
    // Access the projectName via metadata
    const userMetadata = await this.projectStore.loadUserMetadata()
    const [newUserMetadata, projectName] = this.removeSocketFromMeta(
      userMetadata,
      socketId
    )
    if (projectName === "") {
      return
    }
    await this.projectStore.saveUserMetadata(newUserMetadata)

    // Next remove the user info for that project
    let userData = await this.projectStore.loadUserData(projectName)
    userData = this.removeSocketFromUser(userData, socketId)
    await this.projectStore.saveUserData(userData)
  }

  /**
   * Counts the number of currently connected users
   *
   * @param projectName
   */
  public async countUsers(projectName: string): Promise<number> {
    if (this.disable) {
      return 0
    }
    const userData = await this.projectStore.loadUserData(projectName)
    const userToSockets = userData.userToSockets
    return Object.keys(userToSockets).length
  }

  /**
   * Remove all active users, so all counts should be 0
   */
  public async clearUsers(): Promise<void> {
    if (this.disable) {
      return
    }
    // Access the project names from the metadata
    const userMetadata = await this.projectStore.loadUserMetadata()
    const activeProjects = Object.values(userMetadata.socketToProject)

    // Clear the metadata
    await this.projectStore.saveUserMetadata(makeUserMetadata())

    // Remove the user info for each project
    for (const projectName of activeProjects) {
      await this.projectStore.saveUserData(makeUserData(projectName))
    }
  }

  /**
   * Links socket to user and vice versa
   *
   * @param userData
   * @param socketId
   * @param userId
   */
  private addSocketToUser(
    userData: UserData,
    socketId: string,
    userId: string
  ): UserData {
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

    return {
      projectName: userData.projectName,
      socketToUser,
      userToSockets
    }
  }

  /**
   * Unlinks socket and user
   *
   * @param userData
   * @param socketId
   */
  private removeSocketFromUser(userData: UserData, socketId: string): UserData {
    const socketToUser = userData.socketToUser
    const userToSockets = userData.userToSockets

    if (!(socketId in socketToUser)) {
      // Socket has no associated user
      return userData
    }
    // Remove map from socket to user
    const userId = socketToUser[socketId]
    // TODO: Use map for flexible deletion
    // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
    delete socketToUser[socketId]

    // Remove socket from its user's list
    const socketInd = userToSockets[userId].indexOf(socketId)
    if (socketInd > -1) {
      userToSockets[userId].splice(socketInd, 1)
    }

    // Remove the user if it has no sockets left
    if (userToSockets[userId].length === 0) {
      // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
      delete userToSockets[userId]
    }
    return {
      projectName: userData.projectName,
      socketToUser,
      userToSockets
    }
  }

  /**
   * Updates metadata by linking socket to project
   *
   * @param userMetadata
   * @param socketId
   * @param projectName
   */
  private addSocketToMeta(
    userMetadata: UserMetadata,
    socketId: string,
    projectName: string
  ): UserMetadata {
    const socketToProject = userMetadata.socketToProject
    socketToProject[socketId] = projectName
    return { socketToProject }
  }

  /**
   * Updates metadata by removing socket from project
   * Returns updated metadata and the project name if it exists
   *
   * @param userMetadata
   * @param socketId
   */
  private removeSocketFromMeta(
    userMetadata: UserMetadata,
    socketId: string
  ): [UserMetadata, string] {
    const socketToProject = userMetadata.socketToProject
    if (!(socketId in socketToProject)) {
      // Socket has no associated project
      return [userMetadata, ""]
    }
    const projectName = socketToProject[socketId]

    // Remove the socket info from the metadata
    // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
    delete socketToProject[socketId]

    return [{ socketToProject }, projectName]
  }
}

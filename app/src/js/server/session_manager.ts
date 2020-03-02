import { sprintf } from 'sprintf-js'
import Logger from './logger'
import { getRedisSessionManagerKey } from './path'
import { RedisPubSub } from './redis_pub_sub'
import { RedisStore } from './redis_store'
import {
  RegisterMessageType, ServerConfig,
  SessionManagerData, VirtualProjectData, VirtualSessionData } from './types'
import { index2str } from './util'
import { VirtualSession } from './virtual_session'

/**
 * Watches redis and spawns virtual sessions as needed
 */
export class SessionManager {
  /** env variables */
  protected config: ServerConfig
  /** the redis message broker */
  protected subscriber: RedisPubSub
  /** the redis store */
  protected redisStore: RedisStore
  /** the time in between polls that check session activity */
  protected pollTime: number

  constructor (
    config: ServerConfig, subscriber: RedisPubSub, redisStore: RedisStore) {
    this.config = config
    this.subscriber = subscriber
    this.redisStore = redisStore
    this.pollTime = 1000 * 60 * 5 // 5 minutes in ms
  }

  /**
   * Listens to redis changes
   */
  public async listen () {
    // recreate the virtual users
    const managerData = await this.load()
    managerData.projectToTasks =
      this.makeVirtualSessions(managerData.projectToTasks)
    await this.save(managerData)

    // listen for new real users
    this.subscriber.subscribeRegisterEvent(this.handleRegister.bind(this))
  }

  /**
   * Handles registration of new sockets
   */
  public async handleRegister (_channel: string, message: string) {
    const data = JSON.parse(message) as RegisterMessageType

    // create a new virtual user
    const managerData = await this.load()
    managerData.projectToTasks = this.makeVirtualSession(
      managerData.projectToTasks, data.projectName,
      data.taskIndex, data.address)
    await this.save(managerData)
  }

  /**
   * Create a single virtual session
   * And update the session manager data appropriately
   */
  private makeVirtualSession (
    projectToTasks: { [key: string]: VirtualProjectData },
    projectName: string, taskIndex: number, address: string):
    { [key: string]: VirtualProjectData } {
    if (!(projectName in projectToTasks)) {
      projectToTasks[projectName] = { taskToSessions: {} }
    }

    const taskId = index2str(taskIndex)
    const taskToSessions = projectToTasks[projectName].taskToSessions
    // for now, just keep one virtual user per task
    if (!(taskId in taskToSessions) || taskToSessions[taskId].length === 0) {
      const sessionData =
        this.startVirtualSession(projectName, taskIndex, address)
      taskToSessions[taskId] = [sessionData]
    }
    projectToTasks[projectName].taskToSessions = taskToSessions
    return projectToTasks
  }

  /**
   * Create new virtual sessions for each project/task combination in the map
   * And update the session manager data appropriately
   */
  private makeVirtualSessions (
    projectToTasks: { [key: string]: VirtualProjectData }):
    { [key: string]: VirtualProjectData } {
    for (const projectName of Object.keys(projectToTasks)) {
      const taskToSessions = projectToTasks[projectName].taskToSessions
      for (const taskId of Object.keys(taskToSessions)) {
        if (taskToSessions[taskId].length > 0) {
          const sessionData = taskToSessions[taskId][0]
          const newSessionData = this.startVirtualSession(
            projectName, sessionData.taskIndex, sessionData.address)
          taskToSessions[taskId] = [newSessionData]
        }
      }
      projectToTasks[projectName].taskToSessions = taskToSessions
    }
    return projectToTasks
  }

  /**
   * Load the session manager data from redis
   */
  private async load (): Promise<SessionManagerData> {
    const key = getRedisSessionManagerKey()
    const value = await this.redisStore.get(key)
    if (value) {
      return JSON.parse(value)
    } else {
      return { projectToTasks: {} }
    }
  }

  /**
   * Save the session manager data to redis
   */
  private async save (data: SessionManagerData) {
    const key = getRedisSessionManagerKey()
    const value = JSON.stringify(data)
    await this.redisStore.set(key, value)
  }

  /**
   * Create and start a new virtual session
   */
  private startVirtualSession (
    projectName: string, taskIndex: number, address: string):
    VirtualSessionData {
    Logger.info(sprintf('Creating virtual session for project "%s", task %d',
      projectName, taskIndex))

    const sess = new VirtualSession(projectName, taskIndex, address)
    sess.listen()

    const taskId = index2str(taskIndex)
    const pollId = setInterval(async () => {
      await this.checkSessionActivity(sess, projectName, taskId, pollId)
    }, this.pollTime)

    return {
      id: sess.id,
      address,
      taskIndex
    }
  }

  /**
   * Kill the session if no activity since time of last poll
   */
  private async checkSessionActivity (
    session: VirtualSession, projectName: string,
    taskId: string, pollId: NodeJS.Timeout) {
    if (session.actionCount > 0) {
      session.actionCount = 0
    } else {
      session.kill()
      clearInterval(pollId)
      // remove the session from the manager data
      const managerData = await this.load()
      managerData.projectToTasks[projectName].taskToSessions[taskId] = []
      await this.save(managerData)
    }
  }
}

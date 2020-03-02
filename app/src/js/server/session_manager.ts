import { sprintf } from 'sprintf-js'
import Logger from './logger'
import { RedisPubSub } from './redis_pub_sub'
import { RegisterMessageType, ServerConfig } from './types'
import { VirtualSession } from './virtual_session'

/**
 * Watches redis and spawns virtual sessions as needed
 */
export class SessionManager {
  /** env variables */
  protected config: ServerConfig
  /** the redis message broker */
  protected subscriber: RedisPubSub
  /** the task indices that already have virtual sessions for each project */
  protected registeredTasks: { [key: string]: Set<number> }

  constructor (config: ServerConfig, subscriber: RedisPubSub) {
    this.config = config
    this.subscriber = subscriber
    // TODO: store this in redis
    this.registeredTasks = {}
  }

  /**
   * Listens to redis changes
   */
  public listen () {
    this.subscriber.subscribeRegisterEvent(this.handleRegister.bind(this))
  }

  /**
   * Handles registration of new sockets
   */
  public handleRegister (_channel: string, message: string) {
    const data = JSON.parse(message) as RegisterMessageType
    const projectName = data.projectName
    const taskIndex = data.taskIndex
    const address = data.address
    if (!(projectName in this.registeredTasks)) {
      this.registeredTasks[projectName] = new Set<number>()
    }

    if (!this.registeredTasks[projectName].has(taskIndex)) {
      this.registeredTasks[projectName].add(taskIndex)
      this.makeVirtualSession(projectName, taskIndex, address)
    }
  }

  /**
   * Create and start a new virtual session
   */
  private makeVirtualSession (
    projectName: string, taskIndex: number, address: string) {
    Logger.info(sprintf('Creating new virtual session for project \
"%s", task %d', projectName, taskIndex))
    const sess = new VirtualSession(projectName, taskIndex, address)
    sess.listen()
    // TODO: kill the sess if number of sockets in room is 1 (0 + self)
  }
}

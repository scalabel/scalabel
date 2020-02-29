import { RedisPubSub } from './redis_pub_sub'
import { RegisterMessageType, ServerConfig } from './types'
import { index2str } from './util'
import { VirtualSession } from './virtual_session'

/**
 * Watches redis and spawns virtual sessions as needed
 */
export class SessionManager {
  /** env variables */
  protected config: ServerConfig
  /** the redis message broker */
  protected pubSub: RedisPubSub
  /** the task indices that already have virtual sessions for each project */
  protected registeredTasks: { [key: string]: Set<number> }

  constructor (config: ServerConfig, pubSub: RedisPubSub) {
    this.config = config
    this.pubSub = pubSub
    // TODO: store this in redis
    this.registeredTasks = {}
  }

  /**
   * Listens to redis changes
   */
  public listen () {
    this.pubSub.subscribeRegisterEvent(this.handleRegister.bind(this))
  }

  /**
   * Handles registration of new sockets
   */
  public handleRegister (_channel: string, message: string) {
    const data = JSON.parse(message) as RegisterMessageType
    const projectName = data.projectName
    const taskIndex = data.taskIndex
    if (!(projectName in this.registeredTasks)) {
      this.registeredTasks[projectName] = new Set<number>()
    }

    if (!this.registeredTasks[projectName].has(taskIndex)) {
      this.registeredTasks[projectName].add(taskIndex)
      this.makeVirtualSession(projectName, taskIndex)
    }
  }

  /**
   * Create and start a new virtual session
   */
  private makeVirtualSession (projectName: string, taskIndex: number) {
    const sess = new VirtualSession(projectName, index2str(taskIndex))
    sess.listen()

    // TODO: kill the sess if number of sockets in room is 1 (0 + self)
  }
}

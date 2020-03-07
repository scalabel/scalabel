import { BotData } from './types'
import { VirtualSession } from './virtual_session'

/**
 * Manages virtual sessions for a single bot
 */
export class BotUser {
  /** web user id */
  public webId: string
  /** bot user id */
  public botId: string
  /** address for session connections */
  public address: string
  /** list of sessions */
  public sessions: VirtualSession[]

  constructor (botData: BotData) {
    this.webId = botData.webId
    this.botId = botData.botId
    this.address = botData.serverAddress
    this.sessions = []

    /*
     * TODO: should subscribe to redis channel here,
     * so that new session is created when user changes task or project
     */
  }

  /**
   * Make a new session for the user
   */
  public makeSession (projectName: string, taskIndex: number): VirtualSession {
    const sess = new VirtualSession(
      this.botId, this.address, projectName, taskIndex)
    this.sessions.push(sess)
    return sess
  }

  /**
   * Gets the number of actions across all sessions
   */
  public getActionCount (): number {
    let actionCount = 0
    for (const sess of this.sessions) {
      actionCount += sess.actionCount
    }
    return actionCount
  }

  /**
   * Sets action counts to 0 for all sessions
   */

  public resetActionCount () {
    for (const sess of this.sessions) {
      sess.actionCount = 0
    }
  }

  /**
   * Kills all sessions
   */
  public kill () {
    for (const sess of this.sessions) {
      sess.kill()
    }
  }

  /**
   * Wraps instance variables into data object
   */
  public getData (): BotData {
    return {
      botId: this.botId,
      webId: this.webId,
      serverAddress: this.address
    }
  }
}

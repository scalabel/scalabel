/**
 * Watches and modifies state based on what user sessions do
 */
export class VirtualSession {
  /** name of the project */
  protected projectName: string
  /** id of the task */
  protected taskId: string

  constructor (projectName: string, taskId: string) {
    this.projectName = projectName
    this.taskId = taskId

    // TODO: create a socketio client
    // handle registration normally
    // on broadcast of actions, print some log to check
  }

  /**
   * Subscribes to the hub
   */
  public listen () {
    return
  }
}

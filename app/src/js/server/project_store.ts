import { sprintf } from 'sprintf-js'
import { filterXSS } from 'xss'
import { ItemTypeName, LabelTypeName, TrackPolicyType } from '../common/types'
import { makeItemStatus, makeState } from '../functional/states'
import { State, TaskType } from '../functional/types'
import { FileStorage } from './file_storage'
import Logger from './logger'
import { getProjectKey, getSaveDir, getTaskDir, getTaskKey } from './path'
import { RedisStore } from './redis_store'
import { Storage } from './storage'
import { Env, Project } from './types'
import { makeStorage } from './util'

/**
 * Wraps redis cache and storage basic functionality
 * Exposes higher level methods for writing projects, tasks, etc.
 */
export class ProjectStore {
  /**
   * Creates static store then dynamically initializes with async
   */
  public static async make (env: Env) {
    const projectStore = new ProjectStore(env.redisPort, env.redisTimeout,
      env.timeForWrite, env.numActionsForWrite)
    await projectStore.initialize(env.database, env.data)
    return projectStore
  }
  /** the redis store */
  protected redisStore: RedisStore
  /** the permanent storage */
  protected storage: Storage

  constructor (redisPort: number, redisTimeout: number,
               timeForWrite: number, numActionsForWrite: number) {
    this.redisStore = new RedisStore(redisPort, redisTimeout,
      timeForWrite, numActionsForWrite)
    // dummy storage
    this.storage = new FileStorage('')
  }

  /**
   * Initializes values that require async
   */
  public async initialize (database: string, data: string) {
    this.storage = await makeStorage(database, data)
    this.redisStore.initialize(this.storage)
  }

  /**
   * Saves key-value pair
   * If cache is true, saves to redis, which writes back later
   * Otherwise immediately write back to storage
   */
  public async save (key: string, value: string, cache= true) {
    if (cache) {
      await this.redisStore.setExWithReminder(key, value)
    } else {
      await this.storage.save(key, value)
    }
  }

  /**
   * Helper function for saving the state
   */
  public async saveState (state: State, projectName: string, taskId: string) {
    const stringState = JSON.stringify(state)
    const saveDir = getSaveDir(projectName, taskId)
    await this.save(saveDir, stringState)
  }

  /**
   * Loads state from redis if available, else memory
   */
  public async loadState (projectName: string, taskId: string): Promise<State> {
    let state: State

    // first try to load from redis
    const saveDir = getSaveDir(projectName, taskId)
    const redisValue = await this.redisStore.get(saveDir)
    if (redisValue) {
      state = JSON.parse(redisValue)
    } else {
      // otherwise load from storage
      try {
        // first, attempt loading previous submission
        state = await this.loadSavedState(saveDir)
      } catch {
        // if no submissions exist, load from task
        const taskKey = getTaskKey(projectName, taskId)
        state = await this.loadStateFromTask(taskKey)
      }
    }
    return state
  }

  /**
   * Checks whether project name is unique
   */
  public checkProjectName (projectName: string): Promise<boolean> {
    // check if project.json exists in the project folder
    const key = getProjectKey(projectName)
    return this.storage.hasKey(key)
  }

  /**
   * Reads projects from server's disk
   */
  public async getExistingProjects (): Promise<string[]> {
    const files = await this.storage.listKeys('', true)

    // process files into project names
    const names = []
    for (const f of files) {
      // remove any xss vulnerability
      names.push(filterXSS(f))
    }
    return names
  }

  /**
   * Loads the project
   */
  public async loadProject (projectName: string) {
    const key = getProjectKey(projectName)
    const fields = await this.storage.load(key)
    const loadedProject = JSON.parse(fields) as Project
    return loadedProject
  }

  /**
   * Saves the project
   */
  public async saveProject (project: Project) {
    const key = getProjectKey(project.config.projectName)
    const data = JSON.stringify(project, null, 2)
    await this.save(key, data, false)
  }

  /**
   * gets all tasks in project sorted by index
   * @param projectName
   */
  public async getTasksInProject (
    projectName: string): Promise<TaskType[]> {
    const taskPromises: Array<Promise<TaskType>> = []
    const keys = await this.storage.listKeys(getTaskDir(projectName), false)
    // iterate over all keys and load each task asynchronously
    for (const key of keys) {
      taskPromises.push(this.storage.load(key).then((fields) => {
        return JSON.parse(fields) as TaskType
      })
      )
    }
    const tasks = await Promise.all(taskPromises)
    // sort tasks by index
    tasks.sort((a: TaskType, b: TaskType) => {
      return parseInt(a.config.taskId, 10) - parseInt(b.config.taskId, 10)
    })
    return tasks
  }

  /**
   * Saves a list of tasks
   */
  public async saveTasks (tasks: TaskType[]) {
    const promises: Array<Promise<void>> = []
    for (const task of tasks) {
      const key = getTaskKey(task.config.projectName, task.config.taskId)
      const data = JSON.stringify(task, null, 2)
      promises.push(this.save(key, data, false))
    }
    await Promise.all(promises)
  }

    /**
     * Loads a task
     */
  public async loadTask (projectName: string, taskId: string) {
    const key = getTaskKey(projectName, taskId)
    const taskData = await this.storage.load(key)
    const task = JSON.parse(taskData) as TaskType
    return task
  }

  /**
   * Loads the most recent state for the given task. If no such submission throw
   * an error.
   */
  private async loadSavedState (saveDir: string): Promise<State> {
    const keys = await this.storage.listKeys(saveDir, false)
    if (keys.length === 0) {
      throw new Error(sprintf('No submissions found in dir %s', saveDir))
    }
    Logger.info(sprintf('Reading %s\n', keys[keys.length - 1]))
    const fields = await this.storage.load(keys[keys.length - 1])
    return JSON.parse(fields) as State
  }

  /**
   * Loads the state from task.json (created at import)
   * Used for first load
   */
  private async loadStateFromTask (taskKey: string): Promise<State> {
    const fields = await this.storage.load(taskKey)
    const task = JSON.parse(fields) as TaskType
    const state = makeState({ task })

    state.session.itemStatuses = []
    for (const item of state.task.items) {
      const itemStatus = makeItemStatus()
      for (const sensorKey of Object.keys(item.urls)) {
        itemStatus.sensorDataLoaded[Number(sensorKey)] = false
      }
      state.session.itemStatuses.push(itemStatus)
    }

    // TODO: Move this to be in front end after implementing label selector
    switch (state.task.config.itemType) {
      case ItemTypeName.IMAGE:
      case ItemTypeName.VIDEO:
        if (state.task.config.labelTypes.length === 1) {
          switch (state.task.config.labelTypes[0]) {
            case LabelTypeName.BOX_2D:
              state.task.config.policyTypes =
                [TrackPolicyType.LINEAR_INTERPOLATION]
              break
            case LabelTypeName.POLYGON_2D:
              state.task.config.policyTypes =
                [TrackPolicyType.LINEAR_INTERPOLATION]
              break
            case LabelTypeName.CUSTOM_2D:
              state.task.config.labelTypes[0] =
                Object.keys(state.task.config.label2DTemplates)[0]
              state.task.config.policyTypes =
                [TrackPolicyType.LINEAR_INTERPOLATION]
          }
        }
        break
      case ItemTypeName.POINT_CLOUD:
      case ItemTypeName.POINT_CLOUD_TRACKING:
        if (state.task.config.labelTypes.length === 1 &&
            state.task.config.labelTypes[0] === LabelTypeName.BOX_3D) {
          state.task.config.policyTypes =
            [TrackPolicyType.LINEAR_INTERPOLATION]
        }
        break
      case ItemTypeName.FUSION:
        state.task.config.labelTypes =
          [LabelTypeName.BOX_3D, LabelTypeName.PLANE_3D]
        state.task.config.policyTypes = [TrackPolicyType.LINEAR_INTERPOLATION]
        break
    }
    return state
  }
}

import _ from 'lodash'
import { filterXSS } from 'xss'
import { makeItemStatus, makeState } from '../functional/states'
import { Project, StateMetadata, UserData, UserMetadata } from '../types/project'
import { State, TaskType } from '../types/state'
import Logger from './logger'
import * as path from './path'
import { RedisStore } from './redis_store'
import { Storage, StorageStructure } from './storage'
import {
  getPolicy, makeUserData, makeUserMetadata, safeParseJSON } from './util'

/**
 * Wraps redis cache and storage basic functionality
 * Exposes higher level methods for writing projects, tasks, etc.
 */
export class ProjectStore {
  /** the redis store */
  protected redisStore: RedisStore
  /** the permanent storage */
  protected storage: Storage

  constructor (storage: Storage, redisStore: RedisStore) {
    this.storage = storage
    this.redisStore = redisStore
  }

  /**
   * Saves key-value pair
   * If cache is true, saves to redis, which writes back later
   * Otherwise immediately write back to storage
   */
  public async save (
    key: string, value: string,
    cache= false, metadata= '', numActionsSaved= 1) {
    if (cache) {
      await this.redisStore.setExWithReminder(
        key, value, metadata, numActionsSaved)
    } else {
      await this.storage.save(key, value)
    }
  }

  /**
   * Helper function for saving the state
   */
  public async saveState (
    state: State, projectName: string,
    taskId: string, stateMetadata: StateMetadata, numActionsSaved: number) {
    const stringState = JSON.stringify(state)
    const stringMetadata = JSON.stringify(stateMetadata)
    const saveDir = path.getSaveDir(projectName, taskId)
    await this.save(saveDir, stringState, true, stringMetadata, numActionsSaved)
  }

  /**
   * Load the metadata associated with a particular state
   */

  public async loadStateMetadata (
    projectName: string, taskId: string): Promise<StateMetadata> {
    const saveDir = path.getSaveDir(projectName, taskId)
    const metaKey = path.getRedisMetaKey(saveDir)
    let stateMetadata: StateMetadata = {
      projectName,
      taskId,
      actionIds: {}
    }
    if (this.redisStore) {
      const stringStateMetadata = await this.redisStore.get(metaKey)
      if (stringStateMetadata === null) {
        // TODO: principled error handling for state loading
        return stateMetadata
      }
      const loadedMetadata = JSON.parse(stringStateMetadata)
      if (loadedMetadata) {
        stateMetadata = loadedMetadata
      }
    }
    return stateMetadata
  }

  /**
   * Loads state from redis if available, else memory
   */
  public async loadState (projectName: string, taskId: string): Promise<State> {
    let state: State

    // First try to load from redis
    const saveDir = path.getSaveDir(projectName, taskId)
    let redisValue = null
    if (this.redisStore) {
      redisValue = await this.redisStore.get(saveDir)
    }
    if (redisValue) {
      state = safeParseJSON(redisValue)
    } else {
      // Otherwise load from storage
      try {
        // First, attempt loading previous submission
        state = await this.loadSavedState(saveDir)
      } catch {
        // If no submissions exist, load from task
        const taskKey = path.getTaskKey(projectName, taskId)
        state = await this.loadStateFromTask(taskKey)
      }
    }
    return state
  }

  /**
   * Checks whether project name is unique
   */
  public checkProjectName (projectName: string): Promise<boolean> {
    // Check if project.json exists in the project folder
    const key = path.getProjectKey(projectName)
    return this.storage.hasKey(key)
  }

  /**
   * Reads projects from server's disk
   */
  public async getExistingProjects (): Promise<string[]> {
    // All new projects will be in the projects/ sub folder
    const files = await this.storage.listKeys(StorageStructure.PROJECT, true)
    // Remove any xss vulnerability
    const names = files.map((f) => filterXSS(
      f.slice(StorageStructure.PROJECT.length + 1)))
    Logger.info(`Found ${names.length} projects`)
    return names
  }

  /**
   * Loads the project
   */
  public async loadProject (projectName: string) {
    const key = path.getProjectKey(projectName)
    const fields = await this.storage.load(key)
    const loadedProject = safeParseJSON(fields) as Project
    return loadedProject
  }

  /**
   * Saves the project
   */
  public async saveProject (project: Project) {
    const key = path.getProjectKey(project.config.projectName)
    const data = JSON.stringify(project, null, 2)
    await this.save(key, data)
  }

  /**
   * gets all tasks in project sorted by index
   * @param projectName
   */
  public async getTasksInProject (
    projectName: string): Promise<TaskType[]> {
    const taskPromises: Array<Promise<TaskType>> = []
    const taskDir = path.getTaskDir(projectName)
    const keys = await this.storage.listKeys(taskDir, false)
    // Iterate over all keys and load each task asynchronously
    for (const key of keys) {
      taskPromises.push(this.storage.load(key).then((fields) => {
        return safeParseJSON(fields) as TaskType
      })
      )
    }
    const tasks = await Promise.all(taskPromises)
    // Sort tasks by index
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
      const key = path.getTaskKey(task.config.projectName, task.config.taskId)
      const data = JSON.stringify(task, null, 2)
      promises.push(this.save(key, data))
    }
    await Promise.all(promises)
  }

  /**
   * Loads a task
   */
  public async loadTask (
    projectName: string, taskId: string): Promise<TaskType> {
    const key = path.getTaskKey(projectName, taskId)
    const taskData = await this.storage.load(key)
    const task = safeParseJSON(taskData) as TaskType
    return task
  }

  /**
   * Load user data for the project
   * Stored at project/userData.json
   * If it doesn't exist, return default empty object
   */
  public async loadUserData (projectName: string): Promise<UserData> {
    const key = path.getUserKey(projectName)
    const userDataJSON = await this.storage.safeLoad(key)
    if (userDataJSON) {
      return safeParseJSON(userDataJSON) as UserData
    }
    return makeUserData(projectName)
  }

  /**
   * Saves user data for the project
   */
  public async saveUserData (userData: UserData) {
    const projectName = userData.projectName
    const key = path.getUserKey(projectName)
    await this.save(key, JSON.stringify(userData))
  }

  /**
   * Loads metadata shared between all projects
   * Stored at top level, metaData.json
   */
  public async loadUserMetadata (): Promise<UserMetadata> {
    const key = path.getMetaKey()
    const metaDataJSON = await this.storage.safeLoad(key)
    if (!metaDataJSON) {
      return makeUserMetadata()
    }
    // Handle backwards compatability
    const userMetadata = safeParseJSON(metaDataJSON)
    if (_.has(userMetadata, 'socketToProject')) {
      // New code saves as an object, which allows extensions
      return userMetadata
    }
    // Old code saved map of projects directly
    return { socketToProject: userMetadata }
  }

  /**
   * Saves metadata shared between all projects
   */
  public async saveUserMetadata (userMetadata: UserMetadata) {
    const key = path.getMetaKey()
    await this.save(key, JSON.stringify(userMetadata))
  }

  /**
   * Loads the most recent state for the given task. If no such submission throw
   * an error.
   */
  private async loadSavedState (saveDir: string): Promise<State> {
    const keys = await this.storage.listKeys(saveDir, false)
    if (keys.length === 0) {
      throw new Error(`No submissions found in dir ${saveDir}`)
    }
    Logger.info(`Reading ${keys[keys.length - 1]}\n`)
    const fields = await this.storage.load(keys[keys.length - 1])
    return safeParseJSON(fields) as State
  }

  /**
   * Loads the state from task.json (created at import)
   * Used for first load
   */
  private async loadStateFromTask (taskKey: string): Promise<State> {
    const fields = await this.storage.load(taskKey)
    const task = safeParseJSON(fields) as TaskType
    const state = makeState({ task })

    state.session.itemStatuses = []
    for (const item of state.task.items) {
      const itemStatus = makeItemStatus()
      for (const sensorKey of Object.keys(item.urls)) {
        itemStatus.sensorDataLoaded[Number(sensorKey)] = false
      }
      state.session.itemStatuses.push(itemStatus)
    }

    const [trackPolicy, labelTypes] = getPolicy(
      state.task.config.itemType, state.task.config.labelTypes,
      state.task.config.policyTypes, state.task.config.label2DTemplates)
    state.task.config.policyTypes = trackPolicy
    state.task.config.labelTypes = labelTypes

    return state
  }
}

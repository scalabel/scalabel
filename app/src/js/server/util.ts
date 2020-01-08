import { File } from 'formidable'
import * as fs from 'fs-extra'
import * as yaml from 'js-yaml'
import * as path from 'path'
import { sprintf } from 'sprintf-js'
import { filterXSS } from 'xss'
import * as yargs from 'yargs'
import { BundleFile, HandlerUrl, ItemTypeName, LabelTypeName } from '../common/types'
import { State, TaskType } from '../functional/types'
import { FileStorage } from './file_storage'
import Logger from './logger'
import Session from './server_session'
import { Storage } from './storage'
import { CreationForm, DatabaseType, Env, MaybeError } from './types'

/**
 * Initializes global env
 */
export function initEnv () {
  // read the config file name from argv
  const configFlag = 'config'
  const argv = yargs
    .option(configFlag, {
      describe: 'Config file path.'
    })
    .demandOption(configFlag)
    .string(configFlag)
    .argv
  const configDir: string = argv.config

  // load the config into the environment
  const config: Partial<Env> = yaml.load(fs.readFileSync(configDir, 'utf8'))
  Session.setEnv(config)
}

/**
 * Initialize storage
 * @param {string} database: type of storage
 * @param {string} dir: directory name to save at
 */
function makeStorage (
  database: string, dir: string): [MaybeError, Storage] {
  // initialize to default values
  const storage = new FileStorage(dir)
  let err = null

  switch (database) {
    case DatabaseType.S3:
    case DatabaseType.DYNAMO_DB: {
      err = Error(sprintf(
        '%s storage not implemented yet, using file storage', database))
      break
    }
    case DatabaseType.LOCAL: {
      break
    }
    default: {
      err = Error(sprintf(
        '%s is an unknown database format, using file storage', database))
    }
  }

  return [err, storage]
}

/**
 * Initializes global storage
 */
export function initStorage (env: Env) {
  // set up storage
  const [err, storage] = makeStorage(env.database, env.data)
  Logger.error(err)
  Session.setStorage(storage)
}

/**
 * Builds creation form
 * With empty arguments, default is created
 */
export function makeCreationForm (
  projectName = '', itemType = '', labelType = '',
  pageTitle = '', taskSize = 0, instructions = '', demoMode = false
): CreationForm {
  const form: CreationForm = {
    projectName, itemType, labelType, pageTitle,
    instructions, taskSize, demoMode
  }
  return form
}

/**
 * Reads projects from server's disk
 */
export function getExistingProjects (): Promise<string[]> {
  return Session.getStorage().listKeys('', true).then(
    (files: string[]) => {
      // process files into project names
      const names = []
      for (const f of files) {
        // remove any xss vulnerability
        names.push(filterXSS(f))
      }
      return names
    }
  )
}

/**
 * Gets name of json file with project data
 */
export function getProjectKey (projectName: string) {
  // name/project.json
  return path.join(projectName, 'project')
}

/**
 * Gets name of json file with task data
 */
export function getTaskKey (projectName: string, taskId: string): string {
  // name/tasks/000001.json
  return path.join(projectName, 'tasks', taskId)
}

/**
 * Gets name of submission directory for a given task
 * @param projectName
 * @param task
 */
export function getSavedKey (projectName: string, taskId: string): string {
  return path.join(projectName, 'saved', taskId)
}

/**
 * Converts index into a filename of size 6 with
 * trailing zeroes
 */
export function index2str (index: number) {
  return index.toString().padStart(6, '0')
}

/**
 * Checks whether project name is unique
 */
export function checkProjectName (project: string): Promise<boolean> {
  // check if project.json exists in the project folder
  const key = getProjectKey(project)
  return Session.getStorage().hasKey(key)
}

/**
 * Check whether form file exists
 */
export function formFileExists (file: File | undefined): boolean {
  return (file !== undefined && file.size !== 0)
}

/**
 * Gets the handler url for the project
 */
export function getHandlerUrl (itemType: string, labelType: string): string {
  switch (itemType) {
    case ItemTypeName.IMAGE:
      return HandlerUrl.LABEL
    case ItemTypeName.VIDEO:
      if (labelType === LabelTypeName.BOX_2D
        || labelType === LabelTypeName.POLYGON_2D
        || labelType === LabelTypeName.CUSTOM_2D) {
        return HandlerUrl.LABEL
      }
      return HandlerUrl.INVALID
    case ItemTypeName.POINT_CLOUD:
    case ItemTypeName.POINT_CLOUD_TRACKING:
      if (labelType === LabelTypeName.BOX_3D) {
        return HandlerUrl.LABEL
      }
      return HandlerUrl.INVALID
    case ItemTypeName.FUSION:
      if (labelType === LabelTypeName.BOX_3D) {
        return HandlerUrl.LABEL
      }
      return HandlerUrl.INVALID
    default:
      return HandlerUrl.INVALID
  }
}

/**
 * Get the bundle file for the project
 */
export function getBundleFile (labelType: string): string {
  // depends on redux progress
  if (labelType === LabelTypeName.TAG || labelType === LabelTypeName.BOX_2D) {
    return BundleFile.V2
  } else {
    return BundleFile.V1
  }
}

/**
 * Get whether tracking is on
 * and change item type accordingly
 */
export function getTracking (itemType: string): [string, boolean] {
  switch (itemType) {
    case ItemTypeName.VIDEO:
      return [ItemTypeName.IMAGE, true]
    case ItemTypeName.POINT_CLOUD_TRACKING:
      return [ItemTypeName.POINT_CLOUD, true]
    case ItemTypeName.FUSION:
      return [ItemTypeName.FUSION, true]
    default:
      return [itemType, false]
  }
}

/**
 * gets all tasks in project sorted by index
 * @param projectName
 */
export async function getTasksInProject (
  projectName: string): Promise<TaskType[]> {
  const storage = Session.getStorage()
  const taskPromises: Array<Promise<TaskType>> = []
  const keys = await storage.listKeys(path.join(projectName, 'tasks'), false)
  // iterate over all keys and load each task asynchronously
  for (const key of keys) {
    taskPromises.push(storage.load(key).then((fields) => {
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
 * Loads the most recent state for the given task. If no such submission throw
 * an error.
 * @param stateKey
 */
export async function loadSavedState (projectName: string, taskId: string):
  Promise<State> {
  const storage = Session.getStorage()
  const keys = await storage.listKeys(getSavedKey(projectName, taskId), false)
  if (keys.length === 0) {
    throw new Error('No submissions found for task number ${taskId}')
  }
  Logger.info(sprintf('Reading %s\n', keys[keys.length - 1]))
  const fields = await storage.load(keys[keys.length - 1])
  return JSON.parse(fields) as State
}

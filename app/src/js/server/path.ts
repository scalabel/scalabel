import moment from 'moment'
import { join } from 'path'
import { sprintf } from 'sprintf-js'
import { BotData } from './types'
import { index2str } from './util'

/**
 * Converts task id to name of the room for that id
 * If sync is off, room is separated by sessionId
 */
export function getRoomName (
  projectName: string, taskId: string,
  sync: boolean, sessionId = ''): string {
  if (sync) {
    return sprintf('project%s-task%s', projectName, taskId)
  } else {
    return sprintf('project%s-task%s-session%s',
      projectName, taskId, sessionId)
  }
}

/**
 * Get formatted timestamp
 */
export function getNow (): string {
  return moment().format('YYYY-MM-DD_HH-mm-ss')
}

/**
 * Creates path for a timestamped file
 */
export function getFileKey (filePath: string): string {
  return sprintf('%s/%s', filePath, getNow())
}

/**
 * Creates a temporary directory for tests
 */
export function getTestDir (testName: string): string {
  return sprintf('%s-%s', testName, getNow())
}

/**
 * Gets path for user data
 */
export function getUserKey (project: string): string {
  return sprintf('%s/userData', project)
}

/**
 * Gets path for meta data on user
 */
export function getMetaKey (): string {
  return 'metaData'
}

/* path to html file directories, relative to js
 * note that ../ corresponds to index.html
 */
export const HTMLDirectories: string[] =
  ['../control', '../annotation', '../']

/**
 * Converts relative (to js) path into absolute path
 */
export function getAbsoluteSrcPath (relativePath: string) {
  return join(__dirname, relativePath)
}

/**
 * Get name for export download
 * @param {string} projectName
 */
export function getExportName (projectName: string): string {
  return sprintf('%s_export_%s.json', projectName, getNow())
}

/**
 * Get redis key for associated metadata
 * this means data that doesn't get written back to storage
 */
export function getRedisMetaKey (key: string) {
  return sprintf('%s:meta', key)
}

/**
 * Get redis key for reminder metadata
 */
export function getRedisReminderKey (key: string) {
  return sprintf('%s:reminder', key)
}

/**
 * Convert redis metadata or reminder key to redis base key
 */
export function getRedisBaseKey (metadataKey: string) {
  return metadataKey.split(':')[0]
}

/**
 * Gets the redis key used by bots for the task
 */
export function getRedisBotKey (botData: BotData) {
  const projectName = botData.projectName
  const taskId = index2str(botData.taskIndex)
  return sprintf('%s:%s:botKey', projectName, taskId)
}

/**
 * The name of the set of bot user keys
 */
export function getRedisBotSet () {
  return 'redisBotSetName'
}

/**
 * Gets key of file with project data
 */
export function getProjectKey (projectName: string) {
  return join(projectName, 'project')
}

/**
 * Gets key of file with task data
 */
export function getTaskKey (projectName: string, taskId: string): string {
  return join(getTaskDir(projectName), taskId)
}

/**
 * Gets directory with task data for project
 */
export function getTaskDir (projectName: string): string {
  return join(projectName, 'tasks')
}

/**
 * Gets name of submission directory for a given task
 * @param projectName
 * @param task
 */
export function getSaveDir (projectName: string, taskId: string): string {
  return join(projectName, 'saved', taskId)
}

/**
 * API route path
 */
export const API_PATH = '/api'

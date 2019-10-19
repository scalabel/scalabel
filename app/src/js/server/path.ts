import * as moment from 'moment'
import * as path from 'path'
import { sprintf } from 'sprintf-js'

/**
 * Converts task id to name of the room for that id
 * If sync is off, room is separated by session
 */
export function roomName (
  projectName: string, taskId: string,
  sessionId: string, sync: boolean): string {
  if (sync) {
    return sprintf('project%s-task%s', projectName, taskId)
  } else {
    return sprintf('project%s-task%s-session%s',
      projectName, taskId, sessionId)
  }
}

/**
 * Builds the path for loading and saving data
 * Same directory is used for individual mode and sync mode
 */
export function getPath (
  dataDir: string, projectName: string, taskId: string): string {
  return sprintf('%s/%s/saved/%s/',
    dataDir, projectName, taskId)
}

/**
 * Saves a timestamped file
 */
export function getFile (filePath: string) {
  const today = moment().format('YYYY-MM-DD_hh-mm-ss')
  return sprintf('%s/%s.json', filePath, today)
}

/* path to html file directoriess, relative to js */
export const HTMLDirectories: string[] = ['../control', '../annotation']

/**
 * Converts relative (to js) path into absolute path
 */
export function getAbsoluteSrcPath (relativePath: string) {
  return path.join(__dirname, relativePath)
}

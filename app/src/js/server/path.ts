import moment from 'moment'
import * as path from 'path'
import { sprintf } from 'sprintf-js'

/**
 * Converts task id to name of the room for that id
 * If sync is off, room is separated by sessionId
 */
export function roomName (
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
 * Builds the path for loading and saving data
 * Same directory is used for individual mode and sync mode
 */
export function getPath (
  dataDir: string, projectName: string, taskId: string): string {
  return sprintf('%s/%s/saved/%s/',
    dataDir, projectName, taskId)
}

/**
 * Get formatted timestamp
 */
export function getNow (): string {
  return moment().format('YYYY-MM-DD_hh-mm-ss')
}

/**
 * Creates path for a timestamped file
 */
export function getFileKey (filePath: string) {
  return sprintf('%s/%s', filePath, getNow())
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
  return path.join(__dirname, relativePath)
}

/**
 * Get name for export download
 * @param {string} projectName
 */
export function getExportName (projectName: string): string {
  return sprintf('%s_export_%s.json', projectName, getNow())
}

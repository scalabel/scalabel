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
 * Get formatted timestamp
 */
export function getNow (): string {
  return moment().format('YYYY-MM-DD_HH-mm-ss')
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

/**
 * Get redis key for associated metadata
 * this means data that doesn't get written back to storage
 */
export function redisMetaKey (key: string) {
  return sprintf('%s^meta', key)
}

/**
 * Get redis key for reminder metadata
 */
export function redisReminderKey (key: string) {
  return sprintf('%s^reminder', key)
}

/**
 * Get redis base key from a metadata key
 */
export function redisBaseKey (metadataKey: string) {
  return metadataKey.split('^')[0]
}

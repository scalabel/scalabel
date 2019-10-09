import * as moment from 'moment'
import { sprintf } from 'sprintf-js'

/**
 * Converts task id to name of the room for that id
 */
export function roomName (projectName: string, taskId: string): string {
  return sprintf('project%stask%s', projectName, taskId)
}

/**
 * Builds the path for sync data
 */
export function getPath (
  dataDir: string, projectName: string, taskId: string): string {
  return sprintf('%s/%s/saved/%s/',
    dataDir, projectName, taskId)
}

/**
 * Saves a timestamped file
 */
export function getFile (path: string) {
  const today = moment().format('YYYY-MM-DD_hh-mm-ss')
  return sprintf('%s/%s.json', path, today)
}

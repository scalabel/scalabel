import { sprintf } from 'sprintf-js'
import { ConfigType, State } from '../functional/types'

/**
 * Load the task config
 */
export function getConfig (state: State): ConfigType {
  return state.task.config
}

/**
 * Get link to the dashboard
 */
export function getDashboardLink (state: State): string {
  const config = getConfig(state)
  return sprintf('/vendor?project_name=%s', config.projectName)
}

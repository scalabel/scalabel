import { sprintf } from 'sprintf-js'
import { ConfigType } from '../functional/types'
import Session from './session'

/**
 * Contain the single source for URLs
 */
const path = {

  /**
   * Convenient function to get state config
   * @return {ConfigType}
   */
  getConfig (): ConfigType {
    return Session.getState().config
  },

  /**
   * Get the URL of the vendor dashboard
   * @return {string}
   */
  vendorDashboard (): string {
    return sprintf('/vendor?project_name=%s', path.getConfig().projectName)
  }
}

export default path

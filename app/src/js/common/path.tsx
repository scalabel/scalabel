import Session from './session';
import {ConfigType} from '../functional/types';
import {sprintf} from 'sprintf-js';

/**
 * Contain the single source for URLs
 */
class Path {
  public session: typeof Session;

  /**
   * Initialize the state
   */
  constructor() {
    this.session = Session;
  }

  /**
   * Convenient function to get state config
   * @return {ConfigType}
   */
  public getConfig(): ConfigType {
    return Session.getState().config;
  }

  /**
   * @return {string}
   */
  public vendorDashboard(): string {
    return sprintf('/vendor?project_name=%s', this.getConfig().projectName);
  }
}

export default new Path();

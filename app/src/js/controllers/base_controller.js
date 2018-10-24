import {BaseViewer} from '../viewers/base_viewer';
import Session from '../common/session';
import type {StateType} from '../functional/types';

/**
 * Basic controller
 * If there is no temporary object or algorithm involved, this is usually enough
 */
export class BaseController {
  viewers: Array<BaseViewer>;

  /**
   * initialize internal states
   */
  constructor() {
    this.viewers = [];
  }

  /**
   * Set the connected viewer
   * @param {BaseViewer} viewer
   */
  addViewer(viewer: BaseViewer): void {
    this.viewers.push(viewer);
  }

  /**
   * Callback of redux store
   */
  onStateUpdated(): void {
    for (let viewer of this.viewers) {
      viewer.updateState(Session.getState());
    }
  }

  /**
   * Dispatch actions from controllers
   * @param {Object} action
   */
  dispatch(action: Object): void {
    Session.dispatch(action);
  }

  /**
   * Wrapper function for session getState
   * @return {StateType}
   */
  getState(): StateType {
    return Session.getState();
  }
}

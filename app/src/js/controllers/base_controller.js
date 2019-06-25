/* :: import {BaseViewer} from '../viewers/base_viewer'; */
import Session from '../common/session_single';
import type {State} from '../functional/types';

/**
 * Basic controller
 * If there is no temporary object or algorithm involved, this is usually enough
 */
export class BaseController {
  /* :: viewers: Array<BaseViewer>; */

  /**
   * initialize internal states
   */
  constructor() {
    this.viewers = [];
  }

  /**
   * Set the connected viewer
   * @param {BaseViewer} viewer: reference to corresponding viewer
   */
  addViewer(viewer: BaseViewer): void {
    this.viewers.push(viewer);
  }

  /**
   * Callback of redux store
   */
  onStateUpdated(): void {
    for (let viewer of this.viewers) {
      viewer.updateState(this.getState());
    }
  }

  /**
   * Callback of fast store update
   */
  onFastStateUpdated(): void {
    for (let viewer of this.viewers) {
      viewer.updateFastState(this.getFastState());
    }
  }

  /**
   * Dispatch actions from controllers
   * @param {Object} action: action returned by action creator
   */
  dispatch(action: Object): void {
    Session.dispatch(action);
  }

  /**
   * Wrapper function for session getState
   * @return {State}
   */
  getState(): State {
    return Session.getState();
  }

  /**
   * Wrapper function for session getFastState
   * @return {State}
   */
  getFastState(): State {
    return Session.getFastState();
  }

  /**
   * mouseUp callback
   * @param {Object} ignoredEvent: mouse event
   */
  mouseUp(ignoredEvent: Object): void {}
}

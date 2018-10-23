import {BaseViewer} from '../viewers/base_viewer';
import Session from '../common/session';

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
}

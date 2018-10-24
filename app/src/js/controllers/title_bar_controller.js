import * as types from '../actions/action_types';
import {BaseController} from './base_controller';

/**
 * TitleBarController provides callback functions for TitleBarViewer
 */
export class TitleBarController extends BaseController {
  /**
   * Go to the previous Item
   */
  goToPreviousItem() {
    let index = this.getState().current.item;
    this.dispatch({
      type: types.GO_TO_ITEM,
      index: index - 1,
    });
  }

  /**
   * Go to the next Item
   */
  goToNextItem() {
    let index = this.getState().current.item;
    this.dispatch({
      type: types.GO_TO_ITEM,
      index: index + 1,
    });
  }

  /**
   * Save the current state to the server
   */
  save() {
    let state = this.getState();
    let xhr = new XMLHttpRequest();
    xhr.open('POST', './postSaveV2');
    xhr.send(JSON.stringify(state));
  }
}

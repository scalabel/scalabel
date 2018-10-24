import * as types from '../actions/action_types';
import Session from '../common/session';
import {BaseController} from './base_controller';

/**
 * TitleBarController provides callback functions for TitleBarViewer
 */
export class TitleBarController extends BaseController {
  /**
   * Go to the previous Item
   */
  static goToPreviousItem() {
    let index = Session.getState().current.item;
    Session.dispatch({
      type: types.GO_TO_ITEM,
      index: index - 1,
    });
  }

  /**
   * Go to the next Item
   */
  static goToNextItem() {
    let index = Session.getState().current.item;
    Session.dispatch({
      type: types.GO_TO_ITEM,
      index: index + 1,
    });
  }

  /**
   * Save the current state to the server
   */
  static save() {
    let state = Session.getState();
    let xhr = new XMLHttpRequest();
    xhr.open('POST', './postSaveV2');
    xhr.send(JSON.stringify(state));
  }
}

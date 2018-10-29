import * as types from '../actions/action_types';
import {BaseController} from './base_controller';

/**
 * TitleBarController provides callback functions for TitleBarViewer
 */
export class Box2dController extends BaseController {
  /**
   * Dummy draw function to test viewer when mouse down
   * Dispatch new box2d action
   * @param {Object} ignoredEvent: mouse event
   */
  mouseUp(ignoredEvent: Object) {
    let state = this.getState();
    // dispatch a newBox dummy action here to test Box2dViewer
    this.dispatch({
      type: types.NEW_IMAGE_BOX2D_LABEL,
      itemId: state.current.item,
      optionalAttributes: {x: 0, y: 0, w: 70, h: 35},
    });
  }
}

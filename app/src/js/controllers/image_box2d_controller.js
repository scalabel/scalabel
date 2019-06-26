import * as types from '../action/types';
import {BaseController} from './base_controller';

// Constants
// const States = Object.freeze({
//   FREE: 0, RESIZE: 1, MOVE: 2,
// });

/**
 * TitleBarController provides callback functions for TitleBarViewer
 */
export class Box2DController extends BaseController {
  /**
   * Dummy draw function to test viewer when mouse down
   * Dispatch new box2d action
   * @param {Object} event
   */
  mouseUp(event: Object) {
    let state = this.getState();
    // dispatch a newBox dummy action here to test Box2dViewer
    this.dispatch({
      type: types.NEW_IMAGE_BOX2D_LABEL,
      itemId: state.current.item,
      optionalAttributes: {x: event.x, y: event.y, w: 70, h: 35},
    });
  }
}

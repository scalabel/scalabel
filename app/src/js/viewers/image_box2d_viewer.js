import {BaseViewer2D} from './image_base_viewer';
import {BaseController} from '../controllers/base_controller';
import type {RectType} from '../functional/types';

/**
 * Image viewer Class
 */
export class Box2dViewer extends BaseViewer2D {
  /**
   * @param {BaseController} controller
   * @constructor
   */
  constructor(controller: BaseController) {
    super(controller, 'label_canvas');

    // $FlowFixMe
    let self = this;
    this.canvas.addEventListener('mouseup', function(e) {
      self.controller.mouseUp(e);
    });
  }

  /**
   * Get list of shapes
   * @return {Array<RectType>}
   */
  getRects() {
      let index = this.state.current.item;
      let items = this.state.items[index];
      let labels = items.labels;
      let rects = [];
      for (let labelId of labels) {
        let label = this.state.labels[labelId];
        if (label.shapes && label.shapes.length > 0) {
          rects.push(this.state.shapes[label.shapes[0]]);
        }
      }
      return rects;
  }

  /**
   * Draw a single box
   * @param {RectType} rect
   */
  drawRect(rect: RectType) {
      this.context.save();
      let [x, y] = this.toCanvasCoords([rect.x, rect.y]);
      let [w, h] = this.toCanvasCoords([rect.w, rect.h], false);
      this.context.lineWidth = 2 * this.UP_RES_RATIO;
      // this.context.strokeStyle = strokeStyle;
      this.context.strokeRect(x, y, w, h);
      this.context.restore();
  }

  /**
   * Redraw the label canvas
   * @return {boolean}: whether redraw is successful
   */
  redraw() {
    if (!super.redraw()) {
      return false;
    }
    super.redraw();
    let rects = this.getRects();
    for (let rect of rects) {
        this.drawRect(rect);
    }
    return true;
  }
}

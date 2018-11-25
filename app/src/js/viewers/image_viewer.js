import {BaseViewer2D} from './image_base_viewer';
/* :: import {BaseController} from '../controllers/base_controller'; */
import Session from '../common/session';

/**
 * Image viewer Class
 */
export class ImageViewer extends BaseViewer2D {
  /**
   * @param {BaseController} controller: reference to controller
   * @constructor
   */
  constructor(controller/* : BaseController */) {
    super(controller, 'image-canvas', '', false);
  }
  /**
   * Redraw the image canvas.
   * @return {boolean}: whether redraw is successful
   */
  redraw(): boolean {
    // TODO: should support lazy drawing
    if (!super.redraw()) {
      return false;
    }
    let item = this.getCurrentItem();
    let image = Session.images[item.index];
    // draw stuff
    this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.context.drawImage(image, 0, 0, image.width, image.height,
      0, 0, this.canvas.width, this.canvas.height);
    return true;
  }
}



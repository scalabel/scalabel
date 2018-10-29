import {BaseViewer2D} from './image_base_viewer';
import {BaseController} from '../controllers/base_controller';
import Session from '../common/session';

/**
 * Image viewer Class
 */
export class ImageViewer extends BaseViewer2D {
  /**
   * @param {BaseController} controller
   * @constructor
   */
  constructor(controller: BaseController) {
    super(controller, 'image-canvas');
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
    // update the padding box
    let padBox = this._getPadding();
    // draw stuff
    this.context.clearRect(0, 0, padBox.w, padBox.h);
    this.context.drawImage(image, 0, 0, image.width, image.height,
      padBox.x, padBox.y, padBox.w, padBox.h);
    return true;
  }
}



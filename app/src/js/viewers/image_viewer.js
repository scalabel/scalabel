import {ImageBaseViewer} from './image_base_viewer';

/**
 * Image viewer Class
 */
export class ImageViewer extends ImageBaseViewer {
  /**
   * @param {Object} store
   * @param {Array<Image>} images
   * @constructor
   */
  constructor(store: Object, images: Array<Image>) {
    super(store, images, 'image_canvas');
  }
  /**
   * @param {number} index: item index
   * Redraw the image canvas.
   */
  redraw() {
    let index = this.getActiveItem();
    let image = this.images[index];
    // update the padding box
    this.padBox = this._getPadding();
    // draw stuff
    this.ctx.clearRect(0, 0, this.padBox.w,
      this.padBox.h);
    this.ctx.drawImage(image, 0, 0, image.width, image.height,
      this.padBox.x, this.padBox.y, this.padBox.w, this.padBox.h);
  }
}



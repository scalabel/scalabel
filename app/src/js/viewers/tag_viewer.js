import {BaseViewer2D} from './image_base_viewer';
import {sprintf} from 'sprintf-js';
import {BaseController} from '../controllers/base_controller';

/**
 * Tag viewer class
 */
export class TagViewer extends BaseViewer2D {
  /**
   * @param {BaseController} controller
   * @constructor
   */
  constructor(controller: BaseController) {
    super(controller, 'tag_canvas');
  }

  /**
   * Get tags of current active item
   * @return {*}
   */
  getActiveTags() {
    let item = this.getCurrentItem();
    let attributes = {};
    let label = this.state.labels[item.labels[0]];
    if (label) {
      attributes = label.attributes;
    }
    return attributes;
  }

  /**
   * Redraw the image canvas.
   * @return {boolean}
   */
  redraw(): boolean {
    if (!super.redraw()) {
      return false;
    }
    // preparation
    let padBox = this._getPadding();
    let activeTags = this.getActiveTags();
    this.context.font = '24px Arial';
    let abbr = [];
    let attributes = this.state.config.attributes;
    for (let i = 0; i < attributes.length; i++) {
      let key = attributes[i].name;
      if (activeTags && activeTags[key]) {
        abbr.push(sprintf('  %s: %s', attributes[i].name,
                          attributes[i].values[activeTags[key]]));
      }
    }
    this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.context.fillStyle = 'lightgrey';
    this.context.globalAlpha = 0.3;
    this.context.fillRect(padBox.x + 10, padBox.y + 10, 250,
      (abbr.length) ? abbr.length * 35 + 15 : 0);
    this.context.fillStyle = 'red';
    this.context.globalAlpha = 1.0;
    for (let i = 0; i < abbr.length; i++) {
      this.context.fillText(abbr[i],
        padBox.x + 5, padBox.y + 40 + i * 35);
    }
    return true;
  }
}

/* @flow */
import {ImageBaseViewer} from './image_base_viewer';
// $FlowFixMe
import {sprintf} from 'sprintf-js';

/**
 * Tag viewer class
 */
export class TagViewer extends ImageBaseViewer {
  /**
   * @param {Object} store
   * @param {Array<Image>} images
   * @constructor
   */
  constructor(store: Object, images: Array<Image>) {
    super(store, images, 'tag_canvas');
  }

  /**
   * Get tags of current active item
   * @return {*}
   */
  getActiveTags() {
    let state = this.store.getState().present;
    let activeItem = state.current.item;
    let item = state.items[activeItem];
    return item.attributes;
  }

  /**
   * @param {number} index: item index
   * Redraw the image canvas.
   */
  redraw() {
    // preparation
    this.padBox = this._getPadding();
    let activeTags = this.getActiveTags();
    this.ctx.font = '30px Arial';
    let abbr = [];
    let attributes = this.store.getState().present.config.attributes;
    for (let i = 0; i < attributes.length; i++) {
      let key = attributes[i].name;
      if (activeTags && activeTags[key]) {
        abbr.push(sprintf('  %s: %s', attributes[i].tagPrefix,
                          attributes[i].tagSuffixes[activeTags[key]]));
      }
    }
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.fillStyle = 'lightgrey';
    this.ctx.globalAlpha = 0.3;
    this.ctx.fillRect(this.padBox.x + 10, this.padBox.y + 10, 250,
      (abbr.length) ? abbr.length * 35 + 15 : 0);
    this.ctx.fillStyle = 'red';
    this.ctx.globalAlpha = 1.0;
    for (let i = 0; i < abbr.length; i++) {
      this.ctx.fillText(abbr[i],
        this.padBox.x + 5, this.padBox.y + 40 + i * 35);
    }
  }
}

/* @flow */

/**
 * TitleBarViewer class
 * Currently only responsible for counting labels
 */
export class TitleBarViewer {
  store: Object;
  /**
   * @param {Object} store
   * @constructor
   */
  constructor(store: Object) {
    this.store = store;
  }

  /**
   * initialize viewer
   */
  init() {
    if (this.store.getState().present.config.labelType === 'tag') {
      let labelCountTitle = document.getElementById('label_count_title');
      if (labelCountTitle) {
        labelCountTitle.style.visibility = 'hidden';
      }
      let labelCount = document.getElementById('label_count');
      if (labelCount) {
        labelCount.style.visibility = 'hidden';
      }
    } else {
      this.store.subscribe(this.redraw.bind(this));
    }
  }

  /**
   * @param {number} index: item index
   * Redraw the image canvas.
   */
  redraw() {
    let state = this.store.getState().present;
    let currItem = state.current.item;
    let numLabels = state.items[currItem].labels.length;
    let labelCount = document.getElementById('label_count');
    if (labelCount) {
      labelCount.textContent = '' + numLabels;
    }
  }
}

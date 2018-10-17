/* @flow */
import $ from 'jquery';
/**
 * ToolboxViewer class
 * responsible for tool box
 */
export class ToolboxViewer {
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
    let categories: Array<string> =
        this.store.getState().present.config.categories;
    if (categories === null || categories.length === 0) {
      let customCategories = document.getElementById('custom_categories');
      if (customCategories) {
        customCategories.style.visibility = 'hidden';
      }
    }
    this.store.subscribe(this.redraw.bind(this));
  }

  /**
   * Redraw the image canvas.
   */
  redraw() {
    let state = this.store.getState().present;
    let currentItemIndex = state.current.item;
    let currentItem = state.items[currentItemIndex];
    let attributes = currentItem.attributes;
    let attributesMap = state.config.attributes;
    for (let i = 0; i < attributesMap.length; i++) {
      for (let j = 0; j < attributesMap[i].values.length; j++) {
        let selector = $('#custom_attributeselector_' + i + '-' + j);
        let selectedIndices = null;
        if (attributes) {
          selectedIndices = attributes[attributesMap[i].name];
        }
        if (attributes && selectedIndices && selectedIndices[0] === j) {
          selector.addClass('active');
        } else {
          selector.removeClass('active');
        }
      }
    }
  }
}

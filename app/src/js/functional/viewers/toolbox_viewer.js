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
    let currItemId = state.current.item;
    let currItem = state.items[currItemId];
    let attributes = currItem.attributes;
    if (attributes) {
      let attributesMap = state.config.attributes;
      for (let i = 0; i < attributesMap.length; i++) {
        for (let j = 0; j < attributesMap[i].values.length; j++) {
          let selectedIndex = attributes[attributesMap[i].name];
          let selector = $('#custom_attributeselector_' + i + '-' + j);
          if (selectedIndex === j) {
            selector.addClass('active');
          } else {
            selector.removeClass('active');
          }
        }
      }
    } else {
      let attributesMap = state.config.attributes;
      for (let i = 0; i < attributesMap.length; i++) {
        for (let j = 0; j < attributesMap[i].values.length; j++) {
          let selector = $('#custom_attributeselector_' + i + '-' + j);
          selector.removeClass('active');
        }
      }
    }
  }
}

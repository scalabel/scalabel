// @flow
import {sprintf} from 'sprintf-js';
import * as types from '../actions/action_types';
import type {ImageViewerConfigType,
  ItemType, StateType} from '../functional/types';
import _ from 'lodash';
import {makeImageViewerConfig} from '../functional/states';
/* :: import {BaseController} from '../controllers/base_controller'; */
/* :: import {BaseViewer} from '../viewers/base_viewer'; */

import {configureStore, configureFastStore} from './configure_store';

/**
 * Singleton session class
 */
class Session {
  store: Object;
  fastStore: Object; // This store contains the temporary state
  images: Array<Image>;
  itemType: string;
  labelType: string;
  /* :: controllers: Array<BaseController>; */
  /* :: viewers: Array<BaseViewer>; */
  devMode: boolean;

  /**
   * no-op for state initialization
   */
  constructor() {
    this.store = {};
    this.fastStore = configureFastStore();
    this.images = [];
    this.controllers = [];
    this.viewers = [];
    // TODO: make it configurable in the url
    this.devMode = true;
  }

  /**
   * Get current state in store
   * @return {StateType}
   */
  getState(): StateType {
    return this.store.getState().present;
  }

  /**
   * Get the current temporary state. It is for animation rendering.
   * @return {StateType}
   */
  getFastState(): StateType {
    return this.fastStore.getState();
  }

  /**
   * Wrapper for redux store dispatch
   * @param {Object} action: action description
   */
  dispatch(action: Object): void {
    this.store.dispatch(action);
  }

  /**
   * Subscribe all the controllers to the states
   */
  connectControllers() {
    for (let c of this.controllers) {
      this.store.subscribe(c.onStateUpdated.bind(c));
      this.fastStore.subscribe(c.onFastStateUpdated.bind(c));
    }
  }

  /**
   * Initialize state store
   * @param {Object} stateJson: json state from backend
   */
  initStore(stateJson: Object): void {
    this.store = configureStore(stateJson, this.devMode);
    this.store.dispatch({type: types.INIT_SESSION});
    window.store = this.store;
    let state = this.getState();
    this.itemType = state.config.itemType;
    this.labelType = state.config.labelType;
  }

  /**
   * Load all the images in the state
   */
  loadImages(): void {
    let self = this;
    let items = this.getState().items;
    for (let i = 0; i < items.length; i++) {
      let item: ItemType = items[i];
      // Copy item config
      let config: ImageViewerConfigType = {...item.viewerConfig};
      if (_.isEmpty(config)) {
        config = makeImageViewerConfig();
      }
      let url = item.url;
      let image = new Image();
      image.crossOrigin = 'Anonymous';
      self.images.push(image);
      image.onload = function() {
        config.imageHeight = this.height;
        config.imageWidth = this.width;
        self.store.dispatch({type: types.LOAD_ITEM, index: item.index,
          config: config});
      };
      image.onerror = function() {
        alert(sprintf('Image %s was not found.', url));
      };
      image.src = url;
    }
  }
}

export default new Session();

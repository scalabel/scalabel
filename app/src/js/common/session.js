// @flow
import {sprintf} from 'sprintf-js';
import * as types from '../actions/action_types';
import type {ImageViewerConfigType,
  ItemType, StateType} from '../functional/types';
import _ from 'lodash';
import {makeImageViewerConfig} from '../functional/states';
import {configureStore, configureFastStore} from '../redux/configure_store';

import {Window} from './window';

/**
 * Singleton session class
 */
class Session {
  store: Object;
  fastStore: Object; // This store contains the temporary state
  images: Array<Image>;
  itemType: string;
  labelType: string;
  window: Window;
  devMode: boolean;

  /**
   * no-op for state initialization
   */
  constructor() {
    this.store = {};
    this.fastStore = configureFastStore();
    this.images = [];
    // TODO: make it configurable in the url
    this.devMode = true;
    this.window = new Window('labeling-interface');
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
   * Init general labeling session.
   * @param {Object} stateJson: json state from backend
   */
  init(stateJson: Object): void {
    this.initStore(stateJson);
    if (this.itemType === 'image') {
      this.initImage();
    }
    this.window.render();
  }

  /**
   * Initialize tagging interface
   */
  initImage(): void {
    this.loadImages();
  }

  /**
   * Request Session state from the server
   */
  requestState(): void {
    let self = this;
    // collect store from server
    let xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
      if (xhr.readyState === 4) {
        let json = JSON.parse(xhr.response);
        self.init(json);
      }
    };

    // get params from url path. These uniquely identify a SAT.
    let searchParams = new URLSearchParams(window.location.search);
    let taskIndex = parseInt(searchParams.get('task_index'));
    let projectName = searchParams.get('project_name');

    // send the request to the back end
    let request = JSON.stringify({
      'task': {
        'index': taskIndex,
        'projectOptions': {'name': projectName},
      },
    });
    xhr.open('POST', './postLoadAssignmentV2', true);
    xhr.send(request);
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

import {ToolboxController} from '../controllers/toolbox_controller';
import {TitleBarController} from '../controllers/title_bar_controller';
import {ImageViewer} from '../viewers/image_viewer';
import {TagViewer} from '../viewers/tag_viewer';
import {TitleBarViewer} from '../viewers/title_bar_viewer';
import {ToolboxViewer} from '../viewers/toolbox_viewer';
import {sprintf} from 'sprintf-js';
import * as types from '../actions/action_types';
import type {ImageViewerConfigType,
  ItemType, StateType} from '../functional/types';
import _ from 'lodash';
import {makeImageViewerConfig} from '../functional/states';
import {BaseController} from '../controllers/base_controller';
import {BaseViewer} from '../viewers/base_viewer';
import configureStore from '../store/configure_store';

/**
 * Singleton session class
 */
class Session {
  store: Object;
  images: Array<Image>;
  itemType: string;
  labelType: string;
  controllers: Array<BaseController>;
  viewers: Array<BaseViewer>;
  devMode: boolean;

  /**
   * no-op for state initialization
   */
  constructor() {
    this.store = {};
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
   * Wrapper for redux store dispatch
   * @param {Object} action
   */
  dispatch(action: Object): void {
    this.store.dispatch(action);
  }

  /**
   * Wrapper for redux store subscribe
   * @param {function} func: callback function on state chnage.
   */
  subscribe(func: () => void) {
    this.store.subscribe(func);
  }

  /**
   * Intialize state store
   * @param {Object} stateJson
   */
  initStore(stateJson: Object): void {
    this.store = configureStore(stateJson, this.devMode);
    this.store.dispatch({type: types.INIT_SESSION});
    window.store = this.store;
  }

  /**
   * Initialize tagging interface
   * @param {Object} stateJson
   */
  initImageTagging(stateJson: Object): void {
    let self = this;
    this.initStore(stateJson);
    let imageController = new BaseController();
    let tagController = new BaseController();
    let imageViewer: ImageViewer = new ImageViewer(imageController);
    let tagViewer: TagViewer = new TagViewer(tagController);

    // TODO: change this to viewer controller design
    let titleBarController: TitleBarController = new TitleBarController();
    let titleBarViewer: TitleBarViewer = new TitleBarViewer(titleBarController);
    let toolboxController: ToolboxController = new ToolboxController();
    let toolboxViewer: ToolboxViewer = new ToolboxViewer(toolboxController);

    this.controllers = [imageController, tagController, titleBarController,
      toolboxController];
    this.viewers = [imageViewer, tagViewer, titleBarViewer, toolboxViewer];

    for (let c of this.controllers) {
      self.subscribe(c.onStateUpdated.bind(c));
    }

    self.loadImages();

    // TODO: Refactor into a single registration function that takes a list of
    // TODO: viewers and establish direct interactions that do not impact store
    document.getElementsByTagName('BODY')[0].onresize = function() {
      // imageViewer.setScale(imageViewer.scale);
      // imageViewer.redraw();
      // tagViewer.setScale(tagViewer.scale);
      // tagViewer.redraw();
      self.dispatch({type: types.UPDATE_ALL});
    };

    // TODO: move to TitleBarViewer
    let increaseButton = document.getElementById('increase-btn');
    if (increaseButton) {
      increaseButton.onclick = function() {
        self.dispatch({type: types.IMAGE_ZOOM, ratio: 1.05});
      };
    }
    let decreaseButton = document.getElementById('decrease-btn');
    if (decreaseButton) {
      decreaseButton.onclick = function() {
        self.dispatch({type: types.IMAGE_ZOOM, ratio: 1.0 / 1.05});
      };
    }
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

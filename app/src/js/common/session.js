import {Toolbox} from '../controllers/toolbox_controller';
import {PageControl} from '../controllers/page_controller';
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

  /**
   * no-op for state initialization
   */
  constructor() {
    this.store = {};
    this.images = [];
    this.controllers = [];
    this.viewers = [];
  }

  /**
   * Get current state in store
   * @return {StateType}
   */
  getState(): StateType {
    return this.store.getState().present;
  }

  /**
   * Initialize tagging interface
   * @param {Object} store
   */
  initImageTagging(store: Object): void {
    this.store = store;
    // TODO: refactor the general initialization code to a general init()
    store.dispatch({type: types.INIT_SESSION});
    window.store = store;
    new Toolbox(store);
    new PageControl(store);
    let imageController = new BaseController();
    let tagController = new BaseController();
    let imageViewer: ImageViewer = new ImageViewer(imageController);
    let tagViewer: TagViewer = new TagViewer(tagController);

    // TODO: change this to viewer controller design
    let titleBarViewer: TitleBarViewer = new TitleBarViewer(store);
    let toolboxViewer: ToolboxViewer = new ToolboxViewer(store);
    titleBarViewer.init();
    toolboxViewer.init();

    this.controllers = [imageController, tagController];
    this.viewers = [imageViewer, tagViewer];

    for (let c of this.controllers) {
      store.subscribe(c.onStateUpdated.bind(c));
    }

    this.loadImages(store.getState().present.items);

    // TODO: Refactor into a single registration function that takes a list of
    // TODO: viewers and establish direct interactions that do not impact store
    document.getElementsByTagName('BODY')[0].onresize = function() {
      // imageViewer.setScale(imageViewer.scale);
      // imageViewer.redraw();
      // tagViewer.setScale(tagViewer.scale);
      // tagViewer.redraw();
      store.dispatch({type: types.UPDATE_ALL});
    };

    // TODO: move to TitleBarViewer
    let increaseButton = document.getElementById('increase-button');
    if (increaseButton) {
      increaseButton.onclick = function() {
        store.dispatch({type: types.IMAGE_ZOOM, ratio: 1.05});
      };
    }
    let decreaseButton = document.getElementById('decrease-button');
    if (decreaseButton) {
      decreaseButton.onclick = function() {
        store.dispatch({type: types.IMAGE_ZOOM, ratio: 1.0 / 1.05});
      };
    }
  }

  /**
   * Load all the images in the state
   * @param {Array<ItemType>} items
   */
  loadImages(items: Array<ItemType>): void {
    let self = this;
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

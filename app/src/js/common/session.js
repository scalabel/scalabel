// @flow
import {ToolboxController} from '../controllers/toolbox_controller';
import {TitleBarController} from '../controllers/title_bar_controller';
import {ImageViewer} from '../viewers/image_viewer';
import {AssistantViewer} from '../viewers/assistant_viewer';
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
/* :: import {BaseViewer} from '../viewers/base_viewer'; */
import {Box2DViewer} from '../viewers/image_box2d_viewer';
import {Box2DController} from '../controllers/image_box2d_controller';
import {configureStore, configureFastStore} from '../redux/configure_store';

/**
 * Singleton session class
 */
class Session {
  store: Object;
  fastStore: Object; // This store contains the temporary state
  images: Array<Image>;
  itemType: string;
  labelType: string;
  controllers: Array<BaseController>;
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
   * Init image tagging mode
   */
  initImageTagging(): void {
    let imageController = new BaseController();
    let tagController = new BaseController();
    let imageViewer: ImageViewer = new ImageViewer(imageController);
    let assistantViewer: AssistantViewer = new AssistantViewer(imageController);
    let tagViewer: TagViewer = new TagViewer(tagController);

    // TODO: change this to viewer controller design
    let titleBarController: TitleBarController = new TitleBarController();
    let titleBarViewer: TitleBarViewer = new TitleBarViewer(titleBarController);
    let toolboxController: ToolboxController = new ToolboxController();
    let toolboxViewer: ToolboxViewer = new ToolboxViewer(toolboxController);

    this.controllers = [imageController, tagController, titleBarController,
            toolboxController];
    this.viewers = [imageViewer, assistantViewer, tagViewer,
            titleBarViewer, toolboxViewer];
  }

  /**
   * init box2d labeling mode
   */
  initImageBox2DLabeling(): void {
    let imageController = new BaseController();
    let imageViewer: ImageViewer = new ImageViewer(imageController);
    let assistantViewer: AssistantViewer = new AssistantViewer(imageController);
    let box2dController = new Box2DController();
    let box2dViewer: Box2DViewer = new Box2DViewer(box2dController);

    // TODO: change this to viewer controller design
    let titleBarController: TitleBarController = new TitleBarController();
    let titleBarViewer: TitleBarViewer = new TitleBarViewer(titleBarController);
    let toolboxController: ToolboxController = new ToolboxController();
    let toolboxViewer: ToolboxViewer = new ToolboxViewer(toolboxController);

    this.controllers = [imageController, titleBarController,
      toolboxController, box2dController];
    this.viewers = [imageViewer, assistantViewer, titleBarViewer, toolboxViewer,
      box2dViewer];
  }

  /**
   * Initialize tagging interface
   */
  initImageLabeling(): void {
    let self = this;
    if (this.labelType === 'tag') {
      this.initImageTagging();
    } else if (this.labelType === 'box2dv2') {
      this.initImageBox2DLabeling();
    }
    self.connectControllers();
    self.loadImages();

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
   * Init labeling session
   * @param {Object} stateJson: json state from backend
   */
  init(stateJson: Object): void {
    this.initStore(stateJson);

    if (this.itemType === 'image') {
      this.initImageLabeling();
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

import {ImageViewer} from '../viewers/image_viewer';
import {AssistantViewer} from '../viewers/assistant_viewer';
import {TagViewer} from '../viewers/tag_viewer';
import {TitleBarController} from '../controllers/title_bar_controller';
import {TitleBarViewer} from '../viewers/title_bar_viewer';
import {ToolboxController} from '../controllers/toolbox_controller';
import {ToolboxViewer} from '../viewers/toolbox_viewer';
import {Box2DController} from '../controllers/image_box2d_controller';
import {Box2DViewer} from '../viewers/image_box2d_viewer';
import {BaseController} from '../controllers/base_controller';
import * as types from '../actions/action_types';

import Session from './session_single';

/**
 * Init image tagging mode
 */
function initImageTagging(): void {
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

  Session.controllers = [
    imageController, tagController, titleBarController,
    toolboxController];
  Session.viewers = [
    imageViewer, assistantViewer, tagViewer,
    titleBarViewer, toolboxViewer];
}

/**
 * initSingle box2d labeling mode
 */
function initImageBox2DLabeling(): void {
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

  Session.controllers = [
    imageController, titleBarController,
    toolboxController, box2dController];
  Session.viewers = [
    imageViewer, assistantViewer, titleBarViewer, toolboxViewer,
    box2dViewer];
}

/**
 * Initialize tagging interface
 */
function initImageLabelingSingle(): void {
  let self = Session;
  if (self.labelType === 'tag') {
    initImageTagging();
  } else if (self.labelType === 'box2dv2') {
    initImageBox2DLabeling();
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
 * Init labeling session. Each session can only support one single
 * type of labels.
 * @param {Object} stateJson: json state from backend
 */
export function initSingle(stateJson: Object): void {
  Session.initStore(stateJson);

  if (Session.itemType === 'image') {
    initImageLabelingSingle();
  }
}

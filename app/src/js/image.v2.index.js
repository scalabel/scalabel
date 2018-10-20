/* @flow */
import $ from 'jquery';
// $FlowFixMe
import 'bootstrap-material-design';
import configureStore from './store/configure_store';
import {Toolbox} from './controllers/toolbox_controller';
import {ImageViewer} from './viewers/image_viewer';
import {TagViewer} from './viewers/tag_viewer';
import {TitleBarViewer} from './viewers/title_bar_viewer';
import {ToolboxViewer} from './viewers/toolbox_viewer';
import {PageControl} from './controllers/page_controller';
import {sprintf} from 'sprintf-js';
import * as types from './actions/action_types';

declare var labelType: String;
declare var itemType: String;

let frameRate = document.getElementById('frame_rate');
if (frameRate) {
  frameRate.style.display = 'none';
}

let devMode = true;

/**
 * Initialize viewers
 * @param {{}} store
 */
function initViewers(store) {
  // For debug purpose
  if (devMode) {
    window.store = store;
  }
  new Toolbox(store);
  new PageControl(store);
  let images: Array<Image> = [];
  let imageViewer: ImageViewer = new ImageViewer(store, images);
  let tagViewer: TagViewer = new TagViewer(store, images);
  let titleBarViewer: TitleBarViewer = new TitleBarViewer(store);
  let toolboxViewer: ToolboxViewer = new ToolboxViewer(store);
  imageViewer.init();
  tagViewer.init();
  titleBarViewer.init();
  toolboxViewer.init();

  let items = store.getState().present.items;
  for (let i = 0; i < items.length; i++) {
    let url = items[i].url;
    let image = new Image();
    images.push(image);
    image.onload = function() {
      imageViewer.loaded(i);
      tagViewer.loaded(i);
    };
    image.onerror = function() {
      alert(sprintf('Image %s was not found.', url));
    };
    image.src = url;
  }

  store.dispatch({type: types.INIT_SESSION});

  // TODO: Refactor into a single registration function that takes a list of
  // TODO: viewers and establish direct interactions that do not impact store
  document.getElementsByTagName('BODY')[0].onresize = function() {
    imageViewer.setScale(imageViewer.scale);
    imageViewer.redraw();
    tagViewer.setScale(tagViewer.scale);
    tagViewer.redraw();
  };
  let increaseButton = document.getElementById('increase-btn');
  if (increaseButton) {
    increaseButton.onclick = function() {
      imageViewer._incHandler();
      tagViewer._incHandler();
    };
  }
  let decreaseButton = document.getElementById('decrease-btn');
  if (decreaseButton) {
    decreaseButton.onclick = function() {
      imageViewer._decHandler();
      tagViewer._decHandler();
    };
  }
}

$(document).ready(function() {
  // $FlowFixMe
  $('body').bootstrapMaterialDesign();
  // collect item and label class for the session
  // let labelClass;
  // let itemClass;

  if (labelType === 'tag') {
    // labelClass = {};
  }

  if (itemType === 'image') {
    // new Sat(SatImage, labelClass);
    // itemClass = SatImage;
    $('#player-controls').remove();
  }

  // collect store from server
  let store = {};
  let xhr = new XMLHttpRequest();
  xhr.onreadystatechange = function() {
    if (xhr.readyState === 4) {
      let json = JSON.parse(xhr.response);
      store = configureStore(json, devMode);
      initViewers(store);
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
  xhr.open('POST', './postLoadAssignmentV2', false);
  xhr.send(request);
});

$('body').show();

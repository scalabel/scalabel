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

import {ActionCreators} from 'redux-undo';

declare var labelType: String;
declare var itemType: String;

let frameRate = document.getElementById('frame_rate');
if (frameRate) {
  frameRate.style.display = 'none';
}

$(document).ready(function() {
  // $FlowFixMe
  $('body').bootstrapMaterialDesign();
  // Configurations
  let devMode = true;
  // let demoMode = true;

  // collect item and label class for the session
  // let labelClass;
  // let itemClass;

  if (labelType === 'tag') {
    // labelClass = {};
  }

  if (itemType === 'image') {
    // new Sat(SatImage, labelClass);
    // itemClass = SatImage;
    $('#player_controls').remove();
  }

  // collect store from server
  let store = {};
  let xhr = new XMLHttpRequest();
  xhr.onreadystatechange = function() {
    if (xhr.readyState === 4) {
      let json = JSON.parse(xhr.response);
      store = configureStore(json,
        devMode);
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
  // For debug purpose
  if (devMode) {
    window.store = store;
  }
  new Toolbox(store);
  new PageControl(store);
  let images = [];
  let imageViewer: ImageViewer = new ImageViewer(store, images);
  let tagViewer: TagViewer = new TagViewer(store, images);
  let items = store.getState().present.items;
  for (let i = 0; i < items.length; i++) {
    let url = items[i].url;
    let image = new Image();
    image.src = url;
    image.onload = function() {
      imageViewer.loaded(i);
      tagViewer.loaded(i);
    };
    image.onerror = function() {
      alert('Image ' + url + ' was not found.');
    };
    images.push(image);
  }
  let titleBarViewer: TitleBarViewer = new TitleBarViewer(store);
  let toolboxViewer: ToolboxViewer = new ToolboxViewer(store);
  imageViewer.init();
  tagViewer.init();
  titleBarViewer.init();
  toolboxViewer.init();
  // TODO: Refactor into a single registration function that takes a list of
  // TODO: viewers and establish direct interactions that do not impact store
  document.getElementsByTagName('BODY')[0].onresize = function() {
    imageViewer.setScale(imageViewer.scale);
    imageViewer.redraw();
    tagViewer.setScale(tagViewer.scale);
    tagViewer.redraw();
  };
  let increaseBtn = document.getElementById('increase_btn');
  if (increaseBtn) {
    increaseBtn.onclick = function() {
      imageViewer._incHandler();
      tagViewer._incHandler();
    };
  }
  let decreaseBtn = document.getElementById('decrease_btn');
  if (decreaseBtn) {
    decreaseBtn.onclick = function() {
      imageViewer._decHandler();
      tagViewer._decHandler();
    };
  }
  document.addEventListener('keydown', function(e: Object) {
    let eventObj = window.event ? window.event : e;
    if (eventObj.keyCode === 191 && eventObj.shiftKey) {
      // if the user pressed '?':
      $('#keyboard_usage_window').toggle();
    } else if (eventObj.keyCode === 90
      && eventObj.ctrlKey && eventObj.shiftKey) {
      store.dispatch(ActionCreators.redo()); // undo the last action
    } else if (eventObj.keyCode === 90 && eventObj.ctrlKey) {
      store.dispatch(ActionCreators.undo()); // redo the last action
    }
  });
});
$('body').show();

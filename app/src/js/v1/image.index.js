/* global labelType itemType */
import $ from 'jquery';
import {Sat} from './sat';
import {SatImage} from './image';
import {SatVideo} from './video';
import {Box2d} from './box2d';
import {Seg2d} from './seg2d';
import 'bootstrap-material-design';

document.getElementById('frame_rate').style.display = 'none';
let showElementById = function(id) {
  let elem = document.getElementById(id);
  elem.style.visibility = 'visible';
};
let showTemplateById = function(id) {
  let temp = document.getElementById(id);
  let clone = temp.content.cloneNode(true);
  temp.parentNode.appendChild(clone);
};

/**
 * Start a sat session with given labelType and itemType
 * @param {string} labelType
 * @param {string} itemType
 */
export function initSatSession(labelType, itemType) {
  let labelClass;
  if (itemType === 'video') {
    showElementById('player-controls');
    showTemplateById('video_btns');
    showTemplateById('video_usage');
    document.getElementById('div-canvas').style.height = 'calc(100vh - 103px)';
  }
  if (labelType === 'box2d') {
    showElementById('crosshair');
    labelClass = Box2d;
  } else if (labelType === 'segmentation' || labelType === 'lane') {
    labelClass = Seg2d;
    if (labelType === 'segmentation') {
      if (itemType === 'image') {
        showTemplateById('seg_image_btns');
        showTemplateById('seg_image_usage');
      }
      showTemplateById('seg_btns');
      showTemplateById('seg_usage');
    } else if (labelType === 'lane') {
      showTemplateById('lane_usage');
    }
  }
  if (itemType === 'image') {
    new Sat(SatImage, labelClass);
    $('#player-controls').remove();
  }
  if (itemType === 'video') {
    new SatVideo(labelClass);
  }
}

/**
 * Initialize a task by creating the first submission
 * @param {string} labelType
 * @param {string} itemType
 */
export function initTask(labelType, itemType) {
    let labelClass;
    if (labelType === 'box2d') {
      labelClass = Box2d;
    } else if (labelType === 'segmentation' || labelType === 'lane') {
      labelClass = Seg2d;
    }
    let sat;
    if (itemType === 'image') {
      sat = new Sat(SatImage, labelClass);
    }
    if (itemType === 'video') {
      sat = new SatVideo(labelClass);
    }
    sat.save();
  }

$(document).ready(function() {
  $('body').bootstrapMaterialDesign();
  initSatSession(labelType, itemType);
  document.addEventListener('keydown', function(e) {
    let eventObj = window.event ? event : e;
    if (eventObj.keyCode === 191 && eventObj.shiftKey) {
      // if the user pressed '?':
      $('#keyboard_usage_window').toggle();
    }
  });
});

$('body').show();

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
  if (labelType === 'box2d') {
    showElementById('crosshair');
    labelClass = Box2d;
  } else if (labelType === 'segmentation' || labelType === 'lane') {
    labelClass = Seg2d;
    if (labelType === 'segmentation') {
      showTemplateById('seg_btns');
      showTemplateById('seg_usage');
    } else if (labelType === 'lane') {
      showTemplateById('lane_usage');
    }
  }
  if (itemType === 'image') {
    new Sat(SatImage, labelClass);
    $('#player_controls').remove();
  }
  if (itemType === 'video') {
    showElementById('player_controls');
    showTemplateById('video_btns');
    document.getElementById('div_canvas').style.height = 'calc(100vh - 103px)';
    new SatVideo(labelClass);
  }
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

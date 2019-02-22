import $ from 'jquery';
// $FlowFixMe
import 'bootstrap-material-design';
import {initSingle} from '../common/session_single_init';

declare var labelType: String;
declare var itemType: String;

let frameRate = document.getElementById('frame_rate');
if (frameRate) {
  frameRate.style.display = 'none';
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
  let xhr = new XMLHttpRequest();
  xhr.onreadystatechange = function() {
    if (xhr.readyState === 4) {
      let json = JSON.parse(xhr.response);
      initSingle(json);
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

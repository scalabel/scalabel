import Session from './session';
import {Window} from './window';

/**
 * Init general labeling session.
 * @param {Object} stateJson: json state from backend
 */
function initFromJson(stateJson: Object): void {
  Session.initStore(stateJson);
  if (Session.itemType === 'image') {
    initImage();
  }
  Session.window.render();
}

/**
 * Initialize tagging interface
 */
function initImage(): void {
  Session.loadImages();
}

/**
 * Request Session state from the server
 */
export function initSession(): void {
  Session.window = new Window('labeling-interface');

  // collect store from server
  let xhr = new XMLHttpRequest();
  xhr.onreadystatechange = function() {
    if (xhr.readyState === 4) {
      let json = JSON.parse(xhr.response);
      initFromJson(json);
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

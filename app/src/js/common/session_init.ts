import Session from './session';
import Window from './window';
import * as types from '../actions/action_types';

/**
 * Init general labeling session.
 * @param {object} stateJson: json state from backend
 */
function initFromJson(stateJson: object): void {
    Session.initStore(stateJson);
    Session.loadData();
    Session.dispatch({type: types.UPDATE_ALL});
}

/**
 * Request Session state from the server
 * @param {string} containerName: div name for the labeling interface
 */
export function initSession(containerName: string): void {
    // collect store from server
    const xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
            const json = JSON.parse(xhr.response);
            initFromJson(json);
            const window = new Window(containerName);
            window.render();
        }
    };

// get params from url path. These uniquely identify a SAT.
    const searchParams = new URLSearchParams(window.location.search);
    const taskIndex = parseInt(searchParams.get('task_index') as string, 10);
    const projectName = searchParams.get('project_name');

// send the request to the back end
    const request = JSON.stringify({
        task: {
            index: taskIndex,
            projectOptions: {name: projectName}
        }
    });
    xhr.open('POST', './postLoadAssignmentV2', true);
    xhr.send(request);
}

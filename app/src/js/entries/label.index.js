import React from 'react';
import ReactDOM from 'react-dom';
import LabelLayout from '../components/label_layout';
import TitleBar from '../components/title_bar';
import $ from 'jquery';

$(document).ready(function() {
  // collect store from server
  let xhr = new XMLHttpRequest();
  xhr.onreadystatechange = function() {
    if (xhr.readyState === 4) {
      let json = JSON.parse(xhr.response);
      renderInterface(json);
      // Session.init(json);
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

/**
 * Function to render the interface
 * @param {object} json
 */
function renderInterface(json) {
  /* LabelLayout props:
       * titleBar: required
       * center: required
       * leftSidebar1: required
       * leftSidebar2: optional
       * bottomBar: optional
       * rightSidebar1: optional
       * rightSidebar2: optional
       */

  // get all the components
  let titleBar = (
      <TitleBar
      title={json.config.pageTitle}
      instructionLink={json.config.instructionPage}
      dashboardLink={'/vendor?project_name='+json.config.projectName}
    />
  );
  let leftSidebar1 = (<div>1</div>);
  let center = (<div>2</div>);
  let bottomBar = (<div>3</div>);
  let rightSidebar1 = (<div>4</div>);
  let rightSidebar2 = (<div>5</div>);

  let container = document.getElementById('labeling-interface');
  if (container != null) {
    // render the interface
    ReactDOM.render(
        <LabelLayout
            titleBar={titleBar}
            leftSidebar1={leftSidebar1}
            bottomBar={bottomBar}
            center={center}
            rightSidebar1={rightSidebar1}
            rightSidebar2={rightSidebar2}
        />,
        container
    );
  }
}

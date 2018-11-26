import React from 'react';
import ReactDOM from 'react-dom';
import LabelLayout from '../components/label_layout';
import TitleBar from '../components/title_bar';
import Session from './session';
import Path from './path';

/**
 * Manage the whole window
 */
export class Window {
  container: Element;

  /**
   * Window constructor
   * @param {string} containerName: name of the container in HTML to
   * place this window
   */
  constructor(containerName: string) {
    let container = document.getElementById(containerName);
    if (container === null) {
      console.error('Cannot find ' + containerName);
    } else {
      this.container = container;
    }
  }

  /**
   * Function to render the interface
   */
  render() {
    /* LabelLayout props:
         * titleBar: required
         * main: required
         * leftSidebar1: required
         * leftSidebar2: optional
         * bottomBar: optional
         * rightSidebar1: optional
         * rightSidebar2: optional
         */
    let state = Session.getState();

    // get all the components
    let titleBar = (
        <TitleBar
            title={state.config.pageTitle}
            instructionLink={state.config.instructionPage}
            dashboardLink={Path.vendorDashboard()}
        />
    );
    let leftSidebar1 = (<div>1</div>);
    let main = (<div>2</div>);
    let bottomBar = (<div>3</div>);
    let rightSidebar1 = (<div>4</div>);
    let rightSidebar2 = (<div>5</div>);
    // render the interface
    ReactDOM.render(
        <LabelLayout
            titleBar={titleBar}
            leftSidebar1={leftSidebar1}
            bottomBar={bottomBar}
            main={main}
            rightSidebar1={rightSidebar1}
            rightSidebar2={rightSidebar2}
        />,
        this.container,
    );
  }
}

export type WindowType = Window;

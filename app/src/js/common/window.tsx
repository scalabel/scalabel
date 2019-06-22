import React from 'react';
import ReactDOM from 'react-dom';
import LabelLayout from '../components/label_layout';
import TitleBar from '../components/title_bar';
import Session from './session';
import path from './path';
import {ToolBar} from '../components/toolbar';
import MainView from '../components/main_view';
import ImageView from '../components/image_view';
import PointCloudView from '../components/point_cloud_view';
import {sprintf} from 'sprintf-js';

/**
 * Manage the whole window
 */
export default class Window {
  /** The container */
  private container: Element;

  /**
   * Window constructor
   *
   */
  constructor(containerName: string) {
    const container = document.getElementById(containerName);
    if (container) {
      this.container = container;
    } else {
      throw new Error(sprintf("Can't find container %s", containerName));
    }
  }

  /**
   * Function to render the interface
   * @param {string} containerName: name of the container in HTML to
   * place this window
   */
  public render() {
    const state = Session.getState();

    // get all the components
    const titleBar = (
        <TitleBar
            title={state.config.pageTitle}
            instructionLink={state.config.instructionPage}
            dashboardLink={path.vendorDashboard()}
        />
    );
    const leftSidebar1 = (
        <ToolBar
            categories={state.config.categories}
            attributes={state.config.attributes}
            itemType={state.config.itemType}
            labelType={state.config.labelType}
        />
    );

    const views = [];
    if (Session.itemType === 'image') {
      /* FIXME: set correct props */
      views.push(<ImageView key={'imageView'} height={0} width={1}/>);
    } else if (Session.itemType === 'pointcloud') {
      views.push(<PointCloudView key={'pointCloudView'}/>);
    }
    const main = (<MainView views={views} />);
    const bottomBar = (<div>3</div>);
    const rightSidebar1 = (<div>4</div>);
    const rightSidebar2 = (<div>5</div>);
    // render the interface
    if (this.container) {
      ReactDOM.render(
        <LabelLayout
          titleBar={titleBar}
          leftSidebar1={leftSidebar1}
          bottomBar={bottomBar}
          main={main}
          rightSidebar1={rightSidebar1}
          rightSidebar2={rightSidebar2}
        />,
        this.container
      );
    }
  }
}

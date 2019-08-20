import React from 'react'
import Path from '../common/path'
import Session from '../common/session'
import ImageView from './image_view'
import LabelLayout from './label_layout'
import MainView from './main_view'
import PointCloudView from './point_cloud_view'
import TitleBar from './title_bar'
// $FlowFixMe
import { ToolBar } from './toolbar'

/**
 * Manage the whole window
 */

export class Window extends React.Component {
  /**
   * Window constructor
   * @param {object} props: name of the container in HTML to
   * place this window
   */
  constructor (props: object) {
    super(props)
  }

  /**
   * Function to render the interface
   * @return {React.Fragment}
   */
  public render () {
    const state = Session.getState()

    const config = state.task.config

    // get all the components
    const titleBar = (
        <TitleBar
            title={config.pageTitle}
            instructionLink={state.task.config.instructionPage}
            dashboardLink={Path.vendorDashboard()}
        />
    )
    const leftSidebar1 = (
        <ToolBar
            categories={config.categories}
            attributes={config.attributes}
            itemType={config.itemType}
            labelType={config.labelTypes[0]}
        />
    )

    const views = []
    if (Session.itemType === 'image' || Session.itemType === 'video') {
      /* FIXME: set correct props */
      views.push(<ImageView key={'imageView'}/>)
    } else if (Session.itemType === 'pointcloud') {
      views.push(<PointCloudView key={'pointCloudView'}/>)
    }
    const main = (<MainView views={views} />)
    const bottomBar = null
    const rightSidebar1 = null
    const rightSidebar2 = null
    return (
        <LabelLayout
          titleBar={titleBar}
          leftSidebar1={leftSidebar1}
          bottomBar={bottomBar}
          main={main}
          rightSidebar1={rightSidebar1}
          rightSidebar2={rightSidebar2}
        />
    )
  }
}

export default Window

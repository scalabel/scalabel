import React from 'react'
import Path from '../common/path'
import Session from '../common/session'
import Synchronizer from '../common/synchronizer'
import LabelLayout from './label_layout'
import TitleBar from './title_bar'
// $FlowFixMe
import { ToolBar } from './toolbar'
import ViewerContainer from './viewer_container'

interface Props {
  /** global synchronizer for backend */
  synchronizer: Synchronizer
}

/**
 * Manage the whole window
 */
export class Window extends React.Component<Props> {
  /**
   * Window constructor
   * @param {Props} props
   */

  constructor (props: Props) {
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
            autosave={config.autosave}
            synchronizer={this.props.synchronizer}
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

    const main = (<ViewerContainer />)
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

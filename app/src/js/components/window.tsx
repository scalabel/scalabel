import React from 'react'
import { connect } from 'react-redux'
import { StateWithHistory } from 'redux-undo'
import { getConfig } from '../common/selector'
import { Synchronizer } from '../common/synchronizer'
import { ConfigType, State } from '../functional/types'
import LabelLayout from './label_layout'
import TitleBar from './title_bar'
// $FlowFixMe
import { ToolBar } from './toolbar'

interface StateProps {
  /** config variables */
  config: ConfigType
}

interface DependencyProps {
  /** synchronizer for saving */
  synchronizer: Synchronizer
}

type Props = StateProps & DependencyProps

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

    document.addEventListener('contextmenu', (e) => {
      e.preventDefault()
    }, false)

    window.addEventListener('wheel', (e) => {
      e.preventDefault()
    }, { passive: false })
  }

  /**
   * Function to render the interface
   * @return {React.Fragment}
   */
  public render () {
    const config = this.props.config

    // get all the components
    const titleBar = (
        <TitleBar
          synchronizer={this.props.synchronizer}
        />
    )

    const leftSidebar1 = (
        <ToolBar
            categories={config.categories}
            attributes={config.attributes}
            labelType={config.labelTypes[0]}
        />
    )

    const bottomBar = null
    const rightSidebar1 = null
    const rightSidebar2 = null
    return (
        <LabelLayout
          titleBar={titleBar}
          leftSidebar1={leftSidebar1}
          bottomBar={bottomBar}
          rightSidebar1={rightSidebar1}
          rightSidebar2={rightSidebar2}
          key='labelLayout'
        />
    )
  }
}

const mapStateToProps = (state: StateWithHistory<State>): StateProps => {
  return {
    config: getConfig(state)
  }
}
export default connect(mapStateToProps)(Window)

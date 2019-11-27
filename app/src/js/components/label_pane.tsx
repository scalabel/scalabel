import { withStyles } from '@material-ui/styles'
import * as React from 'react'
import SplitPane from 'react-split-pane'
import { updateAll } from '../action/common'
import Session from '../common/session'
import { resizerStyles } from '../styles/split_pane'
import { Component } from './component'
import ViewerContainer, { viewerContainerReactKey } from './viewer_container'

interface ClassType {
  /** class name for resizer */
  resizer: string
}

interface Props {
  /** class names */
  classes: ClassType
  /** pane id */
  pane: number
}

/**
 * Wrapper for SplitPane
 */
class LabelPane extends Component<Props> {
  constructor (props: Props) {
    super(props)
  }

  /** Override render */
  public render () {
    const pane = this.state.user.layout.panes[this.props.pane]
    if (pane.viewerId >= 0) {
      // Leaf, render viewer container
      return (<ViewerContainer
        id={pane.viewerId}
        key={viewerContainerReactKey(pane.viewerId)}
      />)
    }

    if (
      !pane.child1 ||
      !pane.child2 ||
      !(pane.child1 in this.state.user.layout.panes) ||
      !(pane.child2 in this.state.user.layout.panes)
    ) {
      return null
    }
    if (!pane.split) {
      throw new Error('Missing split type')
    }
    if (!pane.primarySize) {
      throw new Error('Missing primary size')
    }

    const child1 = (<StyledLabelPane pane={pane.child1} />)
    const child2 = (<StyledLabelPane pane={pane.child2} />)

    return (
      <SplitPane
        split={pane.split}
        minSize={`${pane.minPrimarySize}%`}
        maxSize={`${pane.maxPrimarySize}%`}
        size={`${pane.primarySize}%`}
        primary={pane.primary}
        onChange={() => Session.dispatch(updateAll())}
        allowResize
        resizerClassName={this.props.classes.resizer}
      >
        {child1}
        {child2}
      </SplitPane>
    )
  }
}

const StyledLabelPane = withStyles(resizerStyles)(LabelPane)

export default StyledLabelPane

import { IconButton } from '@material-ui/core'
import Grid from '@material-ui/core/Grid'
import MenuItem from '@material-ui/core/MenuItem'
import Select from '@material-ui/core/Select'
import CloseIcon from '@material-ui/icons/Close'
import ViewStreamIcon from '@material-ui/icons/ViewStream'
import { withStyles } from '@material-ui/styles'
import * as React from 'react'
import SplitPane from 'react-split-pane'
import { changeViewerConfig, deletePane, splitPane, updatePane } from '../action/common'
import Session from '../common/session'
import * as types from '../common/types'
import { makeDefaultViewerConfig } from '../functional/states'
import { SplitType, ViewerConfigType } from '../functional/types'
import { paneBarStyles, resizerStyles } from '../styles/split_pane'
import { Component } from './component'
import { viewerReactKey } from './drawable_viewer'
import Viewer2D from './viewer2d'
import Viewer3D from './viewer3d'

/** Make drawable viewer based on viewer config */
export function viewerFactory (
  viewerConfig: ViewerConfigType, viewerId: number
) {
  switch (viewerConfig.type) {
    case types.ViewerConfigTypeName.IMAGE:
      return (<Viewer2D id={viewerId} key={viewerReactKey(viewerId)} />)
    case types.ViewerConfigTypeName.POINT_CLOUD:
      return (<Viewer3D id={viewerId} key={viewerReactKey(viewerId)} />)
  }
  return null
}

interface ClassType {
  /** grid */
  viewer_container_bar: string
  /** select */
  select: string
  /** icon */
  icon: string
  /** icon rotated */
  icon90: string
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
      const viewerConfig = this.state.user.viewerConfigs[pane.viewerId]
      const viewerTypeMenu = (
        <Select
          value={viewerConfig.type}
          onChange={(e) => {
            const newConfig = makeDefaultViewerConfig(
              e.target.value as types.ViewerConfigTypeName,
              viewerConfig.pane,
              viewerConfig.sensor
            )
            if (newConfig) {
              Session.dispatch(changeViewerConfig(
                pane.viewerId,
                newConfig
              ))
            }
          }}
          classes={{ select: this.props.classes.select }}
          inputProps={{
            classes: {
              icon: this.props.classes.icon
            }
          }}
        >
          <MenuItem value={types.ViewerConfigTypeName.IMAGE}>Image</MenuItem>
          <MenuItem value={types.ViewerConfigTypeName.POINT_CLOUD}>
            Point Cloud
          </MenuItem>
          <MenuItem value={types.ViewerConfigTypeName.IMAGE_3D}>
            Image 3D
          </MenuItem>
        </Select>
      )

      const verticalSplitButton = (
        <IconButton
          className={this.props.classes.icon90}
          onClick={() => {
            Session.dispatch(splitPane(
              pane.id,
              SplitType.VERTICAL,
              pane.viewerId
            ))
          }}
        >
          <ViewStreamIcon />
        </IconButton>
      )

      const horizontalSplitButton = (
        <IconButton
          className={this.props.classes.icon}
          onClick={() => {
            Session.dispatch(splitPane(
              pane.id,
              SplitType.HORIZONTAL,
              pane.viewerId
            ))
          }}
          edge={'start'}
        >
          <ViewStreamIcon />
        </IconButton>
      )

      const deleteButton = (
        <IconButton
          className={this.props.classes.icon}
          onClick={() => {
            Session.dispatch(deletePane(
              pane.id,
              pane.viewerId
            ))
          }}
          edge={'start'}
        >
          <CloseIcon />
        </IconButton>
      )

      const configBar = (
          <Grid
            justify={'flex-end'}
            container
            direction='row'
            classes={{
              container: this.props.classes.viewer_container_bar
            }}
          >
            {viewerTypeMenu}
            {verticalSplitButton}
            {horizontalSplitButton}
            {deleteButton}
          </Grid>
      )
      // Leaf, render viewer container
      return (
          <div>
            {
              // Disable the config bar for now for 2D images.
              // It will be useful if multiple data sources are present.
              (this.state.task.config.itemType === types.ItemTypeName.IMAGE) ?
                null : configBar
            }
            {viewerFactory(viewerConfig, pane.viewerId)}
          </div>
      )
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

    const child1 = (<StyledLabelPane pane={pane.child1} />)
    const child2 = (<StyledLabelPane pane={pane.child2} />)

    const defaultSize = (pane.primarySize) ? pane.primarySize : '50%'
    return (
      <SplitPane
        split={pane.split}
        defaultSize={defaultSize}
        primary={pane.primary}
        onChange={
          (size) => Session.dispatch(updatePane(pane.id, { primarySize: size }))
        }
        allowResize
        resizerClassName={this.props.classes.resizer}
      >
        {child1}
        {child2}
      </SplitPane>
    )
  }
}

const StyledLabelPane = withStyles((_theme) => ({
  ...resizerStyles(),
  ...paneBarStyles()
}))(LabelPane)

export default StyledLabelPane

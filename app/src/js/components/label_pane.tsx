import { IconButton } from '@material-ui/core'
import Grid from '@material-ui/core/Grid'
import MenuItem from '@material-ui/core/MenuItem'
import Select from '@material-ui/core/Select'
import CloseIcon from '@material-ui/icons/Close'
import ViewStreamIcon from '@material-ui/icons/ViewStream'
import VisibilityIcon from '@material-ui/icons/Visibility'
import VisibilityOffIcon from '@material-ui/icons/VisibilityOff'
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
import HomographyViewer from './homography_viewer'
import Image3DViewer from './image3d_viewer'
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
    case types.ViewerConfigTypeName.IMAGE_3D:
      return (<Image3DViewer id={viewerId} key={viewerReactKey(viewerId)} />)
    case types.ViewerConfigTypeName.HOMOGRAPHY:
      return (<HomographyViewer id={viewerId} key={viewerReactKey(viewerId)} />)
  }
  return null
}

/** convert string to enum */
function viewerTypeFromString (type: string): types.ViewerConfigTypeName {
  switch (type) {
    case types.ViewerConfigTypeName.IMAGE:
      return types.ViewerConfigTypeName.IMAGE
    case types.ViewerConfigTypeName.POINT_CLOUD:
      return types.ViewerConfigTypeName.POINT_CLOUD
    case types.ViewerConfigTypeName.IMAGE_3D:
      return types.ViewerConfigTypeName.IMAGE_3D
    case types.ViewerConfigTypeName.HOMOGRAPHY:
      return types.ViewerConfigTypeName.HOMOGRAPHY
    default:
      return types.ViewerConfigTypeName.UNKNOWN
  }
}

/** Returns whether the config types are compatible */
function viewerConfigTypesCompatible (type1: string, type2: string) {
  let configType1 = viewerTypeFromString(type1)
  let configType2 = viewerTypeFromString(type2)

  if (configType1 === types.ViewerConfigTypeName.IMAGE_3D ||
      configType1 === types.ViewerConfigTypeName.HOMOGRAPHY) {
    configType1 = types.ViewerConfigTypeName.IMAGE
  }

  if (configType2 === types.ViewerConfigTypeName.IMAGE_3D ||
      configType2 === types.ViewerConfigTypeName.HOMOGRAPHY) {
    configType2 = types.ViewerConfigTypeName.IMAGE
  }

  return configType1 === configType2
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

const HIDDEN_UNIT_SIZE = 50

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
          key={`viewerTypeMenu${pane.id}`}
          value={viewerConfig.type}
          onChange={(e) => {
            const newConfig = makeDefaultViewerConfig(
              e.target.value as types.ViewerConfigTypeName,
              viewerConfig.pane,
              -1
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
          <MenuItem
            key={`imageTypeMenuItem${pane.id}`}
            value={types.ViewerConfigTypeName.IMAGE}
          >Image</MenuItem>
          <MenuItem
            key={`pcTypeMenuItem${pane.id}`}
            value={types.ViewerConfigTypeName.POINT_CLOUD}
          >
            Point Cloud
          </MenuItem>
          <MenuItem
            key={`image3dTypeMenuItem${pane.id}`}
            value={types.ViewerConfigTypeName.IMAGE_3D}
          >
            Image 3D
          </MenuItem>
          <MenuItem
            key={`homographyTypeMenuItem${pane.id}`}
            value={types.ViewerConfigTypeName.HOMOGRAPHY}
          >
            Homography
          </MenuItem>
        </Select>
      )

      const viewerIdMenu = (
        <Select
          key={`viewerIdMenu${pane.id}`}
          value={
            viewerConfig.sensor
          }
          onChange={(e) => {
            Session.dispatch(changeViewerConfig(
              pane.viewerId,
              { ...viewerConfig, sensor: e.target.value as number }
            ))
          }}
          classes={{ select: this.props.classes.select }}
          inputProps={{
            classes: {
              icon: this.props.classes.icon
            }
          }}
        >
          {Object.keys(this.state.task.sensors).filter((key) =>
              viewerConfigTypesCompatible(
                this.state.task.sensors[Number(key)].type,
                viewerConfig.type
              )
          ).map((key) =>
            <MenuItem
              key={`viewerId${key}MenuItem${pane.id}`}
              value={Number(key)}
            >{key}</MenuItem>
          )}
        </Select>
      )

      const visibilityButton = (
        <IconButton
          className={this.props.classes.icon}
          onClick={() => {
            Session.dispatch(updatePane(pane.id, { hide: !pane.hide }))
          }}
        >
          {
            (pane.hide) ?
              <VisibilityIcon fontSize='small' /> :
              <VisibilityOffIcon fontSize='small' />
          }
        </IconButton>
      )

      const verticalSplitButton = (
        <IconButton
          key={`verticalSplitButton${pane.id}`}
          className={this.props.classes.icon90}
          onClick={() => {
            Session.dispatch(splitPane(
              pane.id,
              SplitType.VERTICAL,
              pane.viewerId
            ))
          }}
          edge={'start'}
        >
          <ViewStreamIcon fontSize='small' />
        </IconButton>
      )

      const horizontalSplitButton = (
        <IconButton
          key={`horizontalSplitButton${pane.id}`}
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
          <ViewStreamIcon fontSize='small' />
        </IconButton>
      )

      const deleteButton = (
        <IconButton
          key={`deleteButton${pane.id}`}
          className={this.props.classes.icon}
          onClick={() => {
            Session.dispatch(deletePane(
              pane.id,
              pane.viewerId
            ))
          }}
          edge={'start'}
        >
          <CloseIcon fontSize='small' />
        </IconButton>
      )

      const numSensors = Object.keys(this.state.task.sensors).length

      const configBar = (
        <Grid
          key={`paneMenu${pane.id}`}
          justify={'flex-end'}
          container
          direction='row'
          classes={{
            container: this.props.classes.viewer_container_bar
          }}
        >
          <div hidden={pane.hide}>
            {(numSensors > 1) ? viewerTypeMenu : null}
            {(numSensors > 1) ? viewerIdMenu : null}
          </div>
          {visibilityButton}
          <div hidden={pane.hide}>
            {verticalSplitButton}
            {horizontalSplitButton}
          </div>
          {deleteButton}
        </Grid>
      )
      // Leaf, render viewer container
      return (
          <div>
            {configBar}
            <div hidden={pane.hide}>
              {viewerFactory(viewerConfig, pane.viewerId)}
            </div>
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

    const child1 =
      (<StyledLabelPane pane={pane.child1} key={`pane${pane.child1}`}/>)
    const child2 =
      (<StyledLabelPane pane={pane.child2} key={`pane${pane.child2}`}/>)

    const child1State = this.state.user.layout.panes[pane.child1]
    const child2State = this.state.user.layout.panes[pane.child2]

    let defaultSize = (pane.primarySize) ? pane.primarySize : '50%'

    let hiddenSize = HIDDEN_UNIT_SIZE
    if (pane.split) {
      if (pane.split === SplitType.HORIZONTAL) {
        hiddenSize *= pane.numHorizontalChildren + 1
      } else {
        hiddenSize *= pane.numVerticalChildren + 1
      }
    }

    if (pane.hide) {
      defaultSize = '50%'
    } else if (child1State.hide) {
      defaultSize = `${hiddenSize}px`
    } else if (child2State.hide) {
      defaultSize = `calc(100% - ${hiddenSize}px)`
    }

    return (
      <SplitPane
        key={`split${pane.id}`}
        split={pane.split}
        defaultSize={defaultSize}
        primary={pane.primary}
        onChange={
          (size) => Session.dispatch(updatePane(pane.id, { primarySize: size }))
        }
        allowResize
        size={defaultSize}
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

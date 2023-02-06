import { IconButton, List, ListItem } from "@material-ui/core"
import Tooltip from "@mui/material/Tooltip"
import Fade from "@mui/material/Fade"
import Grid from "@material-ui/core/Grid"
import MenuItem from "@material-ui/core/MenuItem"
import Select from "@material-ui/core/Select"
import CloseIcon from "@material-ui/icons/Close"
import ViewStreamIcon from "@material-ui/icons/ViewStream"
import VisibilityIcon from "@material-ui/icons/Visibility"
import VisibilityOffIcon from "@material-ui/icons/VisibilityOff"
import { withStyles } from "@material-ui/styles"
import * as React from "react"
import SplitPane from "../common/split_pane/SplitPane"

import {
  changeViewerConfig,
  deletePane,
  splitPane,
  updatePane
} from "../action/common"
import { dispatch } from "../common/session"
import * as types from "../const/common"
import { makeDefaultViewerConfig } from "../functional/states"
import { getSensorTypes, getMinSensorIds } from "../functional/state_util"
import { paneBarStyles, resizerStyles } from "../styles/split_pane"
import { SplitType, ViewerConfigType } from "../types/state"
import { Component } from "./component"
import { viewerReactKey } from "./drawable_viewer"
import HomographyViewer from "./homography_viewer"
import Image3DViewer from "./image3d_viewer"
import Viewer2D from "./viewer2d"
import Viewer3D from "./viewer3d"
import ViewerRadar from "./viewer_radar"

/**
 * Make drawable viewer based on viewer config
 *
 * @param viewerConfig
 * @param viewerId
 */
export function viewerFactory(
  viewerConfig: ViewerConfigType,
  viewerId: number
): JSX.Element {
  switch (viewerConfig.type) {
    case types.ViewerConfigTypeName.IMAGE:
      return <Viewer2D id={viewerId} key={viewerReactKey(viewerId)} />
    case types.ViewerConfigTypeName.POINT_CLOUD:
      return <Viewer3D id={viewerId} key={viewerReactKey(viewerId)} />
    case types.ViewerConfigTypeName.IMAGE_3D:
      return <Image3DViewer id={viewerId} key={viewerReactKey(viewerId)} />
    case types.ViewerConfigTypeName.HOMOGRAPHY:
      return <HomographyViewer id={viewerId} key={viewerReactKey(viewerId)} />
    case types.ViewerConfigTypeName.RADAR:
      return <ViewerRadar id={viewerId} key={viewerReactKey(viewerId)} />
  }
  return <></>
}

/**
 * convert string to enum
 *
 * @param type
 */
function viewerTypeFromString(type: string): types.ViewerConfigTypeName {
  switch (type) {
    case types.ViewerConfigTypeName.IMAGE:
      return types.ViewerConfigTypeName.IMAGE
    case types.ViewerConfigTypeName.POINT_CLOUD:
      return types.ViewerConfigTypeName.POINT_CLOUD
    case types.ViewerConfigTypeName.IMAGE_3D:
      return types.ViewerConfigTypeName.IMAGE_3D
    case types.ViewerConfigTypeName.HOMOGRAPHY:
      return types.ViewerConfigTypeName.HOMOGRAPHY
    case types.ViewerConfigTypeName.RADAR:
      return types.ViewerConfigTypeName.RADAR
    default:
      return types.ViewerConfigTypeName.UNKNOWN
  }
}

/**
 * Returns whether the config types are compatible
 *
 * @param type1
 * @param type2
 */
function viewerConfigTypesCompatible(type1: string, type2: string): boolean {
  let configType1 = viewerTypeFromString(type1)
  let configType2 = viewerTypeFromString(type2)

  if (
    configType1 === types.ViewerConfigTypeName.IMAGE_3D ||
    configType1 === types.ViewerConfigTypeName.HOMOGRAPHY
  ) {
    configType1 = types.ViewerConfigTypeName.IMAGE
  }

  if (
    configType2 === types.ViewerConfigTypeName.IMAGE_3D ||
    configType2 === types.ViewerConfigTypeName.HOMOGRAPHY
  ) {
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
  /** button */
  button: string
}

interface Props {
  /** class names */
  classes: ClassType
  /** pane id */
  pane: number
}

const HIDDEN_UNIT_SIZE = 50

// TODO: remove this disable
/* eslint-disable max-lines-per-function,max-statements */
/**
 * Wrapper for SplitPane
 */
class LabelPane extends Component<Props> {
  /** Override render */
  public render(): React.ReactNode {
    const pane = this.state.user.layout.panes[this.props.pane]
    if (pane.viewerId >= 0) {
      const viewerConfig = this.state.user.viewerConfigs[pane.viewerId]
      const minSensorIds = getMinSensorIds(this.state)

      const isRadarPane = viewerConfig.type === types.ViewerConfigTypeName.RADAR
      const viewerTypeMenu = (
        <Select
          key={`viewerTypeMenu${pane.id}`}
          value={viewerConfig.type}
          onChange={(e) => {
            const newConfig = makeDefaultViewerConfig(
              e.target.value as types.ViewerConfigTypeName,
              viewerConfig.pane,
              minSensorIds[e.target.value as types.ViewerConfigTypeName]
            )
            if (newConfig !== null) {
              dispatch(changeViewerConfig(pane.viewerId, newConfig))
            }
          }}
          classes={{ select: this.props.classes.select }}
          inputProps={{
            classes: {
              icon: this.props.classes.icon
            }
          }}
        >
          {getSensorTypes(this.state).has(types.ViewerConfigTypeName.IMAGE) &&
          !this.state.task.config.labelTypes.includes(
            types.LabelTypeName.BOX_3D
          ) ? (
            <MenuItem
              key={`imageTypeMenuItem${pane.id}`}
              value={types.ViewerConfigTypeName.IMAGE}
            >
              Image
            </MenuItem>
          ) : undefined}
          {getSensorTypes(this.state).has(
            types.ViewerConfigTypeName.POINT_CLOUD
          ) ? (
            <MenuItem
              key={`pcTypeMenuItem${pane.id}`}
              value={types.ViewerConfigTypeName.POINT_CLOUD}
            >
              Point Cloud
            </MenuItem>
          ) : undefined}

          {getSensorTypes(this.state).has(types.ViewerConfigTypeName.IMAGE) &&
          this.state.task.config.labelTypes.includes(
            types.LabelTypeName.BOX_3D
          ) ? (
            <MenuItem
              key={`image3dTypeMenuItem${pane.id}`}
              value={types.ViewerConfigTypeName.IMAGE_3D}
            >
              Image 3D
            </MenuItem>
          ) : undefined}
          {getSensorTypes(this.state).has(types.ViewerConfigTypeName.IMAGE) &&
          this.state.task.config.labelTypes.includes(
            types.LabelTypeName.BOX_3D
          ) ? (
            <MenuItem
              key={`homographyTypeMenuItem${pane.id}`}
              value={types.ViewerConfigTypeName.HOMOGRAPHY}
            >
              BEV
            </MenuItem>
          ) : undefined}
        </Select>
      )

      const viewerIdMenu = (
        <Select
          key={`viewerIdMenu${pane.id}`}
          value={viewerConfig.sensor}
          onChange={(e) => {
            dispatch(
              changeViewerConfig(pane.viewerId, {
                ...viewerConfig,
                sensor: e.target.value as number
              })
            )
          }}
          classes={{ select: this.props.classes.select }}
          inputProps={{
            classes: {
              icon: this.props.classes.icon
            }
          }}
        >
          {Object.keys(this.state.task.sensors)
            .filter((key) =>
              viewerConfigTypesCompatible(
                this.state.task.sensors[Number(key)].type,
                viewerConfig.type
              )
            )
            .map((key) => (
              <MenuItem
                key={`viewerId${key}MenuItem${pane.id}`}
                value={Number(key)}
              >
                {key}
              </MenuItem>
            ))}
        </Select>
      )

      const visibilityButton = (
        <Tooltip
          key={`visibilityButton${pane.id}`}
          title={`${pane.hide ? "Show pane" : "Hide pane"}`}
          enterDelay={500}
          TransitionComponent={Fade}
          TransitionProps={{ timeout: 600 }}
          arrow
        >
          <IconButton
            className={this.props.classes.icon}
            onClick={() => {
              dispatch(updatePane(pane.id, { hide: !pane.hide }))
            }}
          >
            {pane.hide ? (
              <VisibilityIcon fontSize="small" />
            ) : (
              <VisibilityOffIcon fontSize="small" />
            )}
          </IconButton>
        </Tooltip>
      )

      const verticalSplitButton = (
        <Tooltip
          key={`verticalSplitButton${pane.id}`}
          title="Split vertically"
          enterDelay={500}
          TransitionComponent={Fade}
          TransitionProps={{ timeout: 600 }}
          arrow
        >
          <IconButton
            className={this.props.classes.icon90}
            onClick={() => {
              dispatch(splitPane(pane.id, SplitType.VERTICAL, pane.viewerId))
            }}
          >
            <ViewStreamIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      )

      const horizontalSplitButton = (
        <Tooltip
          key={`horizontalSplitButton${pane.id}`}
          title="Split horizontally"
          enterDelay={500}
          TransitionComponent={Fade}
          TransitionProps={{ timeout: 600 }}
          arrow
        >
          <IconButton
            className={this.props.classes.icon}
            onClick={() => {
              dispatch(splitPane(pane.id, SplitType.HORIZONTAL, pane.viewerId))
            }}
          >
            <ViewStreamIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      )

      const deleteButton = (
        <Tooltip
          key={`deleteButton${pane.id}`}
          title="Delete pane"
          enterDelay={500}
          TransitionComponent={Fade}
          TransitionProps={{ timeout: 600 }}
          arrow
        >
          <IconButton
            className={this.props.classes.icon}
            onClick={() => {
              dispatch(deletePane(pane.id, pane.viewerId))
            }}
          >
            <CloseIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      )

      const numSensors = Object.keys(this.state.task.sensors).length
      const is3dProject = this.state.task.config.labelTypes.includes(
        types.LabelTypeName.BOX_3D
      )
      let paneControl: JSX.Element
      if (is3dProject) {
        paneControl = (
          <List>
            <ListItem dense disableGutters>
              {deleteButton}
            </ListItem>
            <ListItem dense disableGutters>
              <div hidden={pane.hide}>{verticalSplitButton}</div>
            </ListItem>
            <ListItem dense disableGutters>
              <div hidden={pane.hide}>{horizontalSplitButton}</div>
            </ListItem>
            <ListItem dense disableGutters>
              {visibilityButton}
            </ListItem>
          </List>
        )
      } else {
        paneControl = <></>
      }

      const configBar = (
        <Grid
          key={`configMenu${pane.id}`}
          justifyContent={"flex-end"}
          container
          direction="row"
          classes={{
            container: this.props.classes.viewer_container_bar
          }}
        >
          {paneControl}
        </Grid>
      )

      const isImage3d =
        getSensorTypes(this.state).has(types.ViewerConfigTypeName.IMAGE) &&
        this.state.task.config.labelTypes.includes(types.LabelTypeName.BOX_3D)

      let paneBar: JSX.Element
      if (isRadarPane){
        paneBar = <></>
      } else {
        paneBar = (
          <Grid
            key={`paneMenu${pane.id}`}
            justifyContent={"flex-start"}
            alignItems={"stretch"}
            container
            direction="row"
          >
            <div hidden={pane.hide}>
              {numSensors > 1 || isImage3d ? viewerTypeMenu : null}
              {numSensors > 1 ? viewerIdMenu : null}
            </div>
          </Grid>
        )
      }
      
      // Leaf, render viewer container
      return (
        <div>
          {paneBar}
          {configBar}
          <div hidden={pane.hide}>
            {viewerFactory(viewerConfig, pane.viewerId)}
          </div>
        </div>
      )
    }
    if (
      pane.child1 === undefined ||
      pane.child1 === 0 ||
      pane.child2 === undefined ||
      pane.child2 === 0 ||
      !(pane.child1 in this.state.user.layout.panes) ||
      !(pane.child2 in this.state.user.layout.panes)
    ) {
      return null
    }
    if (pane.split === undefined) {
      throw new Error("Missing split type")
    }

    const child1 = (
      <StyledLabelPane pane={pane.child1} key={`pane${pane.child1}`} />
    )
    const child2 = (
      <StyledLabelPane pane={pane.child2} key={`pane${pane.child2}`} />
    )

    const child1State = this.state.user.layout.panes[pane.child1]
    const child2State = this.state.user.layout.panes[pane.child2]

    let defaultSize =
      pane.primarySize !== undefined &&
      pane.primarySize !== 0 &&
      pane.primarySize !== ""
        ? pane.primarySize
        : "50%"

    let hiddenSize = HIDDEN_UNIT_SIZE
    if (pane.split === SplitType.HORIZONTAL) {
      hiddenSize *= pane.numHorizontalChildren + 1
    } else {
      hiddenSize *= pane.numVerticalChildren + 1
    }

    if (pane.hide) {
      defaultSize = "50%"
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
        onChange={(size) =>
          dispatch(updatePane(pane.id, { primarySize: size }))
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

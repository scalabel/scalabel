import { IconButton } from '@material-ui/core'
import Grid from '@material-ui/core/Grid'
import MenuItem from '@material-ui/core/MenuItem'
import Select from '@material-ui/core/Select'
import CloseIcon from '@material-ui/icons/Close'
import ViewStreamIcon from '@material-ui/icons/ViewStream'
import { withStyles } from '@material-ui/styles'
import React from 'react'
import { changeViewerConfig, deletePane, splitPane } from '../action/common'
import Session from '../common/session'
import * as types from '../common/types'
import { makeImage3DViewerConfig, makeImageViewerConfig, makePointCloudViewerConfig } from '../functional/states'
import { ImageViewerConfigType, SplitType, ViewerConfigType } from '../functional/types'
import { viewerContainerStyles } from '../styles/viewer_container'
import ViewerConfigUpdater from '../view_config/viewer_config'
import { Component } from './component'
import ImageViewer from './image_viewer'
import Label2dViewer from './label2d_viewer'
import Label3dViewer from './label3d_viewer'
import PointCloudViewer from './point_cloud_viewer'

/** Generate string to use for react component key */
export function viewerContainerReactKey (id: number) {
  return `viewerContainer${id}`
}

/**
 * Create viewer config based on type
 * @param type
 * @param pane
 * @param sensor
 */
function makeViewerConfig (
  type: types.ViewerConfigTypeName,
  pane: number,
  sensor: number
): ViewerConfigType | null {
  switch (type) {
    case types.ViewerConfigTypeName.IMAGE:
      return makeImageViewerConfig(pane, sensor)
    case types.ViewerConfigTypeName.IMAGE_3D:
      return makeImage3DViewerConfig(pane, sensor)
    case types.ViewerConfigTypeName.POINT_CLOUD:
      return makePointCloudViewerConfig(pane, sensor)
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
  /** container */
  viewer_container: string
}

interface Props {
  /** classes */
  classes: ClassType
  /** id of the viewer, for referencing viewer config in state */
  id: number
}

/**
 * Canvas Viewer
 */
class ViewerContainer extends Component<Props> {
  /** Moveable container */
  private _container: HTMLDivElement | null
  /** viewer config */
  private _viewerConfig?: ViewerConfigType
  /** Manage viewer config */
  private _viewerConfigUpdater: ViewerConfigUpdater

  /** UI handler */
  private _keyDownHandler: (e: KeyboardEvent) => void
  /** UI handler */
  private _keyUpHandler: (e: KeyboardEvent) => void

  /**
   * Constructor
   * @param {Object} props: react props
   */
  constructor (props: Props) {
    super(props)
    this._container = null
    this._viewerConfigUpdater = new ViewerConfigUpdater()

    const state = Session.getState()
    if (this.props.id in state.user.viewerConfigs) {
      this._viewerConfig = state.user.viewerConfigs[this.props.id]
    }

    this._keyDownHandler = this.onKeyDown.bind(this)
    this._keyUpHandler = this.onKeyUp.bind(this)
  }

  /**
   * Run when component mounts
   */
  public componentDidMount () {
    super.componentDidMount()
    document.addEventListener('keydown', this._keyDownHandler)
    document.addEventListener('keyup', this._keyUpHandler)
  }

  /**
   * Run when component unmounts
   */
  public componentWillUnmount () {
    super.componentWillUnmount()
    document.removeEventListener('keydown', this._keyDownHandler)
    document.removeEventListener('keyup', this._keyUpHandler)
  }

  /**
   * Render function
   * @return {React.Fragment} React fragment
   */
  public render () {
    const id = this.props.id
    const viewerConfig = this.state.user.viewerConfigs[this.props.id]
    this._viewerConfig = viewerConfig
    if (viewerConfig && this._container) {
      const viewerType = viewerConfig.type
      if (viewerType === types.ViewerConfigTypeName.IMAGE ||
          types.ViewerConfigTypeName.IMAGE_3D) {
        this._container.scrollTop =
        (viewerConfig as ImageViewerConfigType).displayTop
        this._container.scrollLeft =
          (viewerConfig as ImageViewerConfigType).displayLeft
      }
    }
    this._viewerConfigUpdater.updateState(this.state, this.props.id)

    const views: React.ReactElement[] = []
    if (this._viewerConfig) {
      const config = this._viewerConfig
      switch (config.type) {
        case types.ViewerConfigTypeName.IMAGE:
          views.push(
            <ImageViewer
              key={`imageView${id}`} display={this._container} id={id}
            />
          )
          views.push(
            <Label2dViewer
              key={`label2dView${id}`} display={this._container} id={id}
            />
          )
          break
        case types.ViewerConfigTypeName.POINT_CLOUD:
          views.push(
            <PointCloudViewer
              key={`pointCloudView${id}`} display={this._container} id={id}
            />
          )
          views.push(
            <Label3dViewer
              key={`label3dView${id}`} display={this._container} id={id}
            />
          )
          break
        case types.ViewerConfigTypeName.IMAGE_3D:
          views.push(
            <ImageViewer
              key={`imageView${id}`} display={this._container} id={id}
            />
          )
          views.push(
            <Label3dViewer
              key={`label3dView${id}`} display={this._container} id={id}
            />
          )
          break
      }
    }

    const viewerTypeMenu = (
      <Select
        value={
          (this._viewerConfig) ? this._viewerConfig.type :
            types.ViewerConfigTypeName.UNKNOWN
        }
        onChange={this.handleViewerTypeChange}
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
          if (this._viewerConfig) {
            Session.dispatch(splitPane(
              this._viewerConfig.pane,
              SplitType.VERTICAL,
              this.props.id
            ))
          }
        }}
      >
        <ViewStreamIcon />
      </IconButton>
    )

    const horizontalSplitButton = (
      <IconButton
        className={this.props.classes.icon}
        onClick={() => {
          if (this._viewerConfig) {
            Session.dispatch(splitPane(
              this._viewerConfig.pane,
              SplitType.HORIZONTAL,
              this.props.id
            ))
          }
        }}
      >
        <ViewStreamIcon />
      </IconButton>
    )

    const deleteButton = (
      <IconButton
        className={this.props.classes.icon}
        onClick={() => {
          if (this._viewerConfig) {
            Session.dispatch(deletePane(
              this._viewerConfig.pane,
              this.props.id
            ))
          }
        }}
      >
        <CloseIcon />
      </IconButton>
    )

    const viewerContainerBar = (
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

    return (
      <div>
        {viewerContainerBar}
        <div
          ref={(element) => {
            if (element && this._container !== element) {
              this._container = element
              this._viewerConfigUpdater.setContainer(this._container)
              this.forceUpdate()
            }
          }}
          className={this.props.classes.viewer_container}
          onMouseDown={ (e) => this.onMouseDown(e) }
          onMouseUp={ (e) => this.onMouseUp(e) }
          onMouseMove={ (e) => this.onMouseMove(e) }
          onMouseEnter={ (e) => this.onMouseEnter(e) }
          onMouseLeave={ (e) => this.onMouseLeave(e) }
          onDoubleClick={ (e) => this.onDoubleClick(e) }
          onWheel ={ (e) => this.onWheel(e) }
        >
          {views}
        </div>
      </div>
    )
  }

  /** Handle viewer type select change */
  private handleViewerTypeChange (
    e: React.ChangeEvent<{
      /** Inherited from material ui */
      name?: string;
      /** Inherited from material ui */
      value: unknown
    }>
  ): void {
    if (this._viewerConfig) {
      const newConfig = makeViewerConfig(
        e.target.value as types.ViewerConfigTypeName,
        this._viewerConfig.pane,
        this._viewerConfig.sensor
      )
      if (newConfig) {
        Session.dispatch(changeViewerConfig(
          this.props.id,
          newConfig
        ))
      }
    }
  }

  /**
   * Handle mouse down
   * @param e
   */
  private onMouseDown (e: React.MouseEvent) {
    if (e.button === 2) {
      e.preventDefault()
    }
    this._viewerConfigUpdater.onMouseDown(e.clientX, e.clientY, e.button)
  }

  /**
   * Handle mouse up
   * @param e
   */
  private onMouseUp (_e: React.MouseEvent) {
    this._viewerConfigUpdater.onMouseUp()
  }

  /**
   * Handle mouse move
   * @param e
   */
  private onMouseMove (e: React.MouseEvent) {
    this._viewerConfigUpdater.onMouseMove(e.clientX, e.clientY)
  }

  /**
   * Handle double click
   * @param e
   */
  private onDoubleClick (e: React.MouseEvent) {
    this._viewerConfigUpdater.onDoubleClick(e.clientX, e.clientY)
  }

  /**
   * Handle mouse leave
   * @param e
   */
  private onMouseEnter (_e: React.MouseEvent) {
    Session.activeViewerId = this.props.id
  }

  /**
   * Handle mouse leave
   * @param e
   */
  private onMouseLeave (_e: React.MouseEvent) {
    return
  }

  /**
   * Handle mouse wheel
   * @param e
   */
  private onWheel (e: React.WheelEvent) {
    e.preventDefault()
    this._viewerConfigUpdater.onWheel(e.deltaY)
  }

  /**
   * Handle key down
   * @param e
   */
  private onKeyUp (e: KeyboardEvent) {
    if (Session.activeViewerId === this.props.id) {
      this._viewerConfigUpdater.onKeyUp(e.key)
    }
  }

  /**
   * Handle key down
   * @param e
   */
  private onKeyDown (e: KeyboardEvent) {
    if (Session.activeViewerId === this.props.id) {
      this._viewerConfigUpdater.onKeyDown(e.key)
    }
  }
}

export default withStyles(
  viewerContainerStyles, { withTheme: true }
)(ViewerContainer)

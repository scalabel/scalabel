import React from 'react'
import Session from '../common/session'
import * as types from '../common/types'
import { ImageViewerConfigType, ViewerConfigType } from '../functional/types'
import ViewerConfigUpdater from '../view_config/viewer_config'
import { Component } from './component'
import ImageViewer from './image_viewer'
import Label2dViewer from './label2d_viewer'
import Label3dViewer from './label3d_viewer'
import MouseEventListeners from './mouse_event_listeners'
import PlayerControl from './player_control'
import PointCloudViewer from './point_cloud_viewer'

interface Props {
  /** id of the viewer, for referencing viewer config in state */
  id: number
}

/**
 * Canvas Viewer
 */
class ViewerContainer extends Component<Props> {
  /** Topmost div */
  private _divRef: HTMLDivElement | null
  /** Moveable container */
  private _container: HTMLDivElement | null
  /** viewer config */
  private _viewerConfig?: ViewerConfigType
  /** Manage viewer config */
  private _viewerConfigUpdater: ViewerConfigUpdater

  /** UI handler */
  private _mouseDownHandler: (e: MouseEvent) => void
  /** UI handler */
  private _mouseUpHandler: (e: MouseEvent) => void
  /** UI handler */
  private _mouseMoveHandler: (e: MouseEvent) => void
  /** UI handler */
  private _mouseLeaveHandler: (e: MouseEvent) => void
  /** UI handler */
  private _doubleClickHandler: (e: MouseEvent) => void
  /** UI handler */
  private _wheelHandler: (e: WheelEvent) => void
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
    this._divRef = null
    this._viewerConfigUpdater = new ViewerConfigUpdater()

    this._mouseDownHandler = this.onMouseDown.bind(this)
    this._mouseUpHandler = this.onMouseUp.bind(this)
    this._mouseMoveHandler = this.onMouseMove.bind(this)
    this._mouseLeaveHandler = this.onMouseLeave.bind(this)
    this._doubleClickHandler = this.onDoubleClick.bind(this)
    this._wheelHandler = this.onWheel.bind(this)
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
    let rectDiv: DOMRect | ClientRect | null = null
    if (this._divRef) {
      rectDiv = this._divRef.getBoundingClientRect()
    }

    const id = this.props.id

    const views: React.ReactElement[] = []
    if (this._viewerConfig) {
      const config = this._viewerConfig
      switch (config.type) {
        case types.ViewerConfigType.IMAGE:
          views.push(<ImageViewer key={'imageView'} display={null} id={id} />)
          views.push(
            <Label2dViewer key={'label2dView'} display={null} id={id} />
          )
          break
        case types.ViewerConfigType.POINT_CLOUD:
          views.push(
            <PointCloudViewer key={'pointCloudView'} display={null} id={id} />
          )
          views.push(
            <Label3dViewer key={'label3dView'} display={null} id={id} />
          )
          break
        case types.ViewerConfigType.IMAGE_3D:
          views.push(<ImageViewer key={'imageView'} display={null} id={id} />)
          views.push(
            <Label3dViewer key={'label3dView'} display={null} id={id} />
          )
          break
      }
    }

    let viewsWithProps = views
    if (rectDiv) {
      viewsWithProps = React.Children.map(views, (view) => {
        if (rectDiv) {
          return React.cloneElement(view,
            {
              height: rectDiv.height,
              width: rectDiv.width,
              display: this._container
            })
        } else {
          return React.cloneElement(view, { display: this._container })
        }
      }
      )
    }

    const playerControl = (<PlayerControl key='player-control'
      num_frames={Session.getState().task.items.length}
    />)

    return (
        <div
          ref={(element) => {
            if (element) {
              this._divRef = element
            }
          }}
          style={{
            display: 'block', height: '100%',
            position: 'absolute',
            outline: 'none', width: '100%', background: '#222222'
          }}
        >
          <div
            ref={(element) => {
              if (element) {
                this._container = element
                this._viewerConfigUpdater.setContainer(this._container)
              }
            }}
            style={{
              display: 'block',
              height: 'calc(100% - 20px)',
              top: '10px', left: '10px',
              position: 'absolute',
              overflow: 'hidden',
              outline: 'none',
              width: 'calc(100% - 20px)'
            }}
          >
            <MouseEventListeners
              onMouseDown={this._mouseDownHandler}
              onMouseMove={this._mouseMoveHandler}
              onMouseUp={this._mouseUpHandler}
              onMouseLeave={this._mouseLeaveHandler}
              onDblClick={this._doubleClickHandler}
              onWheel={this._wheelHandler}
            />
            {viewsWithProps}
          </div>
          { playerControl }
        </div >
    )
  }

  /**
   * Run when state is updated
   * @param state
   */
  public componentDidUpdate (): void {
    const viewerConfig = this.state.user.viewerConfigs[this.props.id]
    this._viewerConfig = viewerConfig
    if (viewerConfig && this._container) {
      const viewerType = viewerConfig.type
      if (viewerType === types.ViewerConfigType.IMAGE ||
          types.ViewerConfigType.IMAGE_3D) {
        this._container.scrollTop =
          (viewerConfig as ImageViewerConfigType).displayTop
        this._container.scrollLeft =
          (viewerConfig as ImageViewerConfigType).displayLeft
      }
    }
    this._viewerConfigUpdater.updateState(this.state, this.props.id)
  }

  /**
   * Handle mouse down
   * @param e
   */
  private onMouseDown (e: MouseEvent) {
    this._viewerConfigUpdater.onMouseDown(e)
  }

  /**
   * Handle mouse up
   * @param e
   */
  private onMouseUp (e: MouseEvent) {
    this._viewerConfigUpdater.onMouseUp(e)
  }

  /**
   * Handle mouse move
   * @param e
   */
  private onMouseMove (e: MouseEvent) {
    this._viewerConfigUpdater.onMouseMove(e)
  }

  /**
   * Handle double click
   * @param e
   */
  private onDoubleClick (e: MouseEvent) {
    this._viewerConfigUpdater.onDoubleClick(e)
  }

  /**
   * Handle mouse leave
   * @param e
   */
  private onMouseLeave (_e: MouseEvent) {
    return
  }

  /**
   * Handle mouse wheel
   * @param e
   */
  private onWheel (e: WheelEvent) {
    this._viewerConfigUpdater.onWheel(e)
  }

  /**
   * Handle key down
   * @param e
   */
  private onKeyUp (e: KeyboardEvent) {
    this._viewerConfigUpdater.onKeyUp(e)
  }

  /**
   * Handle key down
   * @param e
   */
  private onKeyDown (e: KeyboardEvent) {
    this._viewerConfigUpdater.onKeyDown(e)
  }
}

export default ViewerContainer

import { IconButton } from '@material-ui/core'
import FindReplaceIcon from '@material-ui/icons/FindReplace'
import ZoomInIcon from '@material-ui/icons/ZoomIn'
import ZoomOutIcon from '@material-ui/icons/ZoomOut'
import { withStyles } from '@material-ui/styles'
import React from 'react'
import { changeViewerConfig } from '../action/common'
import Session from '../common/session'
import * as types from '../common/types'
import { ImageViewerConfigType } from '../functional/types'
import { Vector2D } from '../math/vector2d'
import { viewerStyles } from '../styles/viewer'
import {
  MAX_SCALE,
  MIN_SCALE,
  SCROLL_ZOOM_RATIO
} from '../view_config/image'
import { DrawableViewer, ViewerClassTypes, ViewerProps } from './drawable_viewer'
import ImageCanvas from './image_canvas'
import Label2dCanvas from './label2d_canvas'

interface ClassType extends ViewerClassTypes {
  /** buttons */
  viewer_button: string
}

export interface Viewer2DProps extends ViewerProps {
  /** classes */
  classes: ClassType
}

/**
 * Viewer for images and 2d labels
 */
export class Viewer2D extends DrawableViewer<Viewer2DProps> {
  /**
   * Constructor
   * @param {Object} props: react props
   */
  constructor (props: Viewer2DProps) {
    super(props)
  }

  /**
   * Render function
   * @return {React.Fragment} React fragment
   */
  protected getDrawableComponents () {
    if (this._container && this._viewerConfig) {
      this._container.scrollTop =
        (this._viewerConfig as ImageViewerConfigType).displayTop
      this._container.scrollLeft =
        (this._viewerConfig as ImageViewerConfigType).displayLeft
    }

    const views: React.ReactElement[] = []
    if (this._viewerConfig) {
      views.push(
        <ImageCanvas
          key={`imageCanvas${this.props.id}`}
          display={this._container}
          id={this.props.id}
        />
      )
      views.push(
        <Label2dCanvas
          key={`label2dCanvas${this.props.id}`}
          display={this._container}
          id={this.props.id}
        />
      )
    }

    return views
  }

  /**
   * Render function
   * @return {React.Fragment} React fragment
   */
  protected getMenuComponents () {
    if (this._viewerConfig) {
      const zoomInButton = (
        <IconButton
          onClick={() => {
            if (this._container) {
              const rect = this._container.getBoundingClientRect()
              this.zoom(
                SCROLL_ZOOM_RATIO,
                new Vector2D(rect.width / 2, rect.height / 2)
              )
            }
          }}
          className={this.props.classes.viewer_button}
        >
          <ZoomInIcon />
        </IconButton>
      )
      const zoomOutButton = (
        <IconButton
          onClick={() => {
            if (this._container) {
              const rect = this._container.getBoundingClientRect()
              this.zoom(
                1. / SCROLL_ZOOM_RATIO,
                new Vector2D(rect.width / 2, rect.height / 2)
              )
            }
          }}
          className={this.props.classes.viewer_button}
          edge={'start'}
        >
          <ZoomOutIcon />
        </IconButton>
      )
      const resetZoomButton = (
        <IconButton
          onClick={() => {
            const config = this._viewerConfig as ImageViewerConfigType
            const newConfig = { ...config }
            newConfig.displayLeft = 0
            newConfig.displayTop = 0
            newConfig.viewScale = 1
            Session.dispatch(changeViewerConfig(
              this._viewerId,
              newConfig
            ))
          }}
          className={this.props.classes.viewer_button}
          edge={'start'}
        >
          <FindReplaceIcon />
        </IconButton>
      )
      return [zoomInButton, zoomOutButton, resetZoomButton]
    }
    return []
  }

  /**
   * Handle mouse move
   * @param e
   */
  protected onMouseMove (e: React.MouseEvent) {
    const oldX = this._mX
    const oldY = this._mY
    super.onMouseMove(e)
    if (this._mouseDown && this._container && this._viewerConfig) {
      if (this.isKeyDown(types.Key.META) ||
          this.isKeyDown(types.Key.CONTROL)) {
        const dx = this._mX - oldX
        const dy = this._mY - oldY
        const displayLeft = this._container.scrollLeft - dx
        const displayTop = this._container.scrollTop - dy
        const newConfig = {
          ...this._viewerConfig,
          displayLeft,
          displayTop
        }
        Session.dispatch(changeViewerConfig(
          this._viewerId, newConfig
        ))
      }
    }
  }

  /**
   * Handle double click
   * @param e
   */
  protected onDoubleClick () {
    return
  }

  /**
   * Handle mouse leave
   * @param e
   */
  protected onMouseLeave () {
    return
  }

  /**
   * Handle mouse wheel
   * @param e
   */
  protected onWheel (e: WheelEvent) {
    e.preventDefault()
    if (this._viewerConfig && this._container) {
      if (this.isKeyDown(types.Key.META) ||
          this.isKeyDown(types.Key.CONTROL)) {
        let zoomRatio = SCROLL_ZOOM_RATIO
        if (-e.deltaY < 0) {
          zoomRatio = 1. / zoomRatio
        }
        this.zoom(zoomRatio, new Vector2D(this._mX, this._mY))
      }
    }
  }

  /** Zoom */
  protected zoom (zoomRatio: number, offset: Vector2D) {
    const config = this._viewerConfig as ImageViewerConfigType
    const newScale = config.viewScale * zoomRatio
    const newConfig = { ...config }
    if (newScale >= MIN_SCALE && newScale <= MAX_SCALE) {
      newConfig.viewScale = newScale
    } else {
      zoomRatio = 1
    }
    const displayLeft = zoomRatio * (offset.x + config.displayLeft) -
        offset.x
    const displayTop = zoomRatio * (offset.y + config.displayTop) -
        offset.y
    newConfig.displayLeft = displayLeft
    newConfig.displayTop = displayTop
    Session.dispatch(changeViewerConfig(
      this._viewerId,
      newConfig
    ))
  }
}

export default withStyles(viewerStyles)(Viewer2D)

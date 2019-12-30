import { withStyles } from '@material-ui/styles'
import React from 'react'
import { changeViewerConfig } from '../action/common'
import Session from '../common/session'
import * as types from '../common/types'
import { ImageViewerConfigType } from '../functional/types'
import { viewerStyles } from '../styles/viewer'
import {
  MAX_SCALE,
  MIN_SCALE,
  SCROLL_ZOOM_RATIO
} from '../view_config/image'
import { DrawableViewer, ViewerProps } from './drawable_viewer'
import ImageCanvas from './image_canvas'
import Label2dCanvas from './label2d_canvas'

/**
 * Viewer for images and 2d labels
 */
class Viewer2D extends DrawableViewer {
  /**
   * Constructor
   * @param {Object} props: react props
   */
  constructor (props: ViewerProps) {
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
    if (this._viewerConfig) {
      if (this.isKeyDown(types.Key.META) ||
          this.isKeyDown(types.Key.CONTROL)) {
        let zoomRatio = SCROLL_ZOOM_RATIO
        if (-e.deltaY < 0) {
          zoomRatio = 1. / zoomRatio
        }
        const config = this._viewerConfig as ImageViewerConfigType
        const newScale = config.viewScale * zoomRatio
        const newConfig = { ...config }
        if (newScale >= MIN_SCALE && newScale <= MAX_SCALE) {
          newConfig.viewScale = newScale
        }
        if (this._container) {
          const displayLeft = zoomRatio * (this._mX + config.displayLeft) -
            this._mX
          const displayTop = zoomRatio * (this._mY + config.displayTop) -
            this._mY
          newConfig.displayLeft = displayLeft
          newConfig.displayTop = displayTop
        }
        Session.dispatch(changeViewerConfig(
          this._viewerId,
          newConfig
        ))
      }
    }
  }
}

export default withStyles(viewerStyles)(Viewer2D)

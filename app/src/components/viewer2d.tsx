import { IconButton } from "@material-ui/core"
import FindReplaceIcon from "@material-ui/icons/FindReplace"
import ZoomInIcon from "@material-ui/icons/ZoomIn"
import ZoomOutIcon from "@material-ui/icons/ZoomOut"
import { withStyles } from "@material-ui/styles"
import React from "react"

import { changeViewerConfig } from "../action/common"
import Session from "../common/session"
import * as types from "../const/common"
import { Vector2D } from "../math/vector2d"
import { viewerStyles } from "../styles/viewer"
import { ImageViewerConfigType } from "../types/state"
import { MAX_SCALE, MIN_SCALE, SCROLL_ZOOM_RATIO } from "../view_config/image"
import {
  DrawableViewer,
  ViewerClassTypes,
  ViewerProps
} from "./drawable_viewer"
import ImageCanvas from "./image_canvas"
import Label2dCanvas from "./label2d_canvas"

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
   * Render function
   *
   * @return {React.Fragment} React fragment
   */
  protected getDrawableComponents(): React.ReactElement[] {
    if (this._container !== null && this._viewerConfig !== undefined) {
      this._container.scrollTop = (this
        ._viewerConfig as ImageViewerConfigType).displayTop
      this._container.scrollLeft = (this
        ._viewerConfig as ImageViewerConfigType).displayLeft
    }

    const views: React.ReactElement[] = []
    if (this._viewerConfig !== undefined) {
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
   *
   * @return {React.Fragment} React fragment
   */
  protected getMenuComponents(): JSX.Element[] | [] {
    if (this._viewerConfig !== undefined) {
      const zoomInButton = (
        <IconButton
          key={`zoomIn2dButton${this.props.id}`}
          onClick={() => {
            if (this._container !== null) {
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
          key={`zoomOut2dButton${this.props.id}`}
          onClick={() => {
            if (this._container !== null) {
              const rect = this._container.getBoundingClientRect()
              this.zoom(
                1 / SCROLL_ZOOM_RATIO,
                new Vector2D(rect.width / 2, rect.height / 2)
              )
            }
          }}
          className={this.props.classes.viewer_button}
          edge={"start"}
        >
          <ZoomOutIcon />
        </IconButton>
      )
      const resetZoomButton = (
        <IconButton
          key={`resetZoom2dButton${this.props.id}`}
          onClick={() => {
            const config = this._viewerConfig as ImageViewerConfigType
            const newConfig = { ...config }
            newConfig.displayLeft = 0
            newConfig.displayTop = 0
            newConfig.viewScale = 1
            Session.dispatch(changeViewerConfig(this._viewerId, newConfig))
          }}
          className={this.props.classes.viewer_button}
          edge={"start"}
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
   *
   * @param e
   */
  protected onMouseMove(e: React.MouseEvent): void {
    const oldX = this._mX
    const oldY = this._mY
    super.onMouseMove(e)
    if (
      this._mouseDown &&
      this._container !== null &&
      this._viewerConfig !== undefined
    ) {
      if (this.isKeyDown(types.Key.META) || this.isKeyDown(types.Key.CONTROL)) {
        const dx = this._mX - oldX
        const dy = this._mY - oldY
        const displayLeft = this._container.scrollLeft - dx
        const displayTop = this._container.scrollTop - dy
        const newConfig = {
          ...this._viewerConfig,
          displayLeft,
          displayTop
        }
        Session.dispatch(changeViewerConfig(this._viewerId, newConfig))
      }
    }
  }

  /**
   * Handle double click
   *
   * @param e
   */
  protected onDoubleClick(): void {}

  /**
   * Handle mouse leave
   *
   * @param e
   */
  protected onMouseLeave(): void {}

  /**
   * Handle mouse wheel
   *
   * @param e
   */
  protected onWheel(e: WheelEvent): void {
    e.preventDefault()
    if (this._viewerConfig !== undefined && this._container !== null) {
      if (this.isKeyDown(types.Key.META) || this.isKeyDown(types.Key.CONTROL)) {
        let zoomRatio = SCROLL_ZOOM_RATIO
        if (-e.deltaY < 0) {
          zoomRatio = 1 / zoomRatio
        }
        this.zoom(zoomRatio, new Vector2D(this._mX, this._mY))
      }
    }
  }

  /**
   * Zoom
   *
   * @param zoomRatio
   * @param offset
   */
  protected zoom(zoomRatio: number, offset: Vector2D): void {
    const config = this._viewerConfig as ImageViewerConfigType
    const newScale = config.viewScale * zoomRatio
    const newConfig = { ...config }
    if (newScale >= MIN_SCALE && newScale <= MAX_SCALE) {
      newConfig.viewScale = newScale

      const item = this.state.user.select.item
      const sensor = this.state.user.viewerConfigs[this.props.id].sensor
      const image = Session.images[item][sensor]

      const iw = image.width * newScale
      const ih = image.height * newScale

      if (this._container !== null) {
        const rect = this._container.getBoundingClientRect()

        let displayLeft = zoomRatio * (offset.x + config.displayLeft) - offset.x
        let displayTop = zoomRatio * (offset.y + config.displayTop) - offset.y
        // The difference between the display area and the displayed image in
        // aspect ratio gives rise to blank regions. Expected behavior
        // is zooming to the center when the blank region exists,
        // or zooming to the cursor otherwise.
        if (rect.height / rect.width > ih / iw) {
          // Zoomed height < that of the display area, blanks on top/bottom
          if ((image.width * rect.height) / rect.width > ih) {
            // Set offset to 0
            displayTop = 0
          }
        } else {
          // Zoomed width < that of the display area, blanks on sides
          if ((image.height * rect.width) / rect.height > iw) {
            // Set offset to 0
            displayLeft = 0
          }
        }
        newConfig.displayLeft = displayLeft
        newConfig.displayTop = displayTop
      }

      Session.dispatch(changeViewerConfig(this._viewerId, newConfig))
    }
  }
}

export default withStyles(viewerStyles)(Viewer2D)

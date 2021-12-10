import { withStyles } from "@material-ui/styles"
import * as React from "react"
import * as THREE from "three"

import { changeViewerConfig } from "../action/common"
import Session from "../common/session"
import { IntrinsicCamera } from "../drawable/3d/intrinsic_camera"
import { viewerStyles } from "../styles/viewer"
import {
  HomographyViewerConfigType,
  ImageViewerConfigType
} from "../types/state"
import { SCROLL_ZOOM_RATIO } from "../view_config/image"
import { DrawableViewer, ViewerProps } from "./drawable_viewer"
import HomographyCanvas from "./homography_canvas"
import Label3dCanvas from "./label3d_canvas"

/**
 * Viewer for images and 2d labels
 */
class HomographyViewer extends DrawableViewer<ViewerProps> {
  /** Intrinsic camera */
  private readonly _camera: IntrinsicCamera

  /**
   * Constructor
   *
   * @param {Object} props: react props
   * @param props
   */
  constructor(props: ViewerProps) {
    super(props)
    this._camera = new IntrinsicCamera()
    this._camera.up = new THREE.Vector3(0, -1, 0)
    this._camera.lookAt(new THREE.Vector3(0, 0, 1))
  }

  /** Component update function */
  public componentDidUpdate(): void {
    if (this._viewerConfig !== null) {
      const sensor = this._viewerConfig?.sensor ?? null
      if (sensor !== null) {
        this._camera.intrinsics = this.state.task.sensors[sensor].intrinsics
      }
      if (this._container !== null) {
        const displayRect = this._container.getBoundingClientRect()
        this._camera.width = displayRect.width
        this._camera.height = displayRect.height
        this._camera.intrinsics = {
          focalLength: {
            x: displayRect.width,
            y: displayRect.height
          },
          focalCenter: {
            x: displayRect.width / 2,
            y: displayRect.height / 2
          }
        }
      }

      this._camera.position.set(0, -50, 10)
      this._camera.up = new THREE.Vector3(0, 0, 1)
      this._camera.lookAt(new THREE.Vector3(0, 2, 20))

      this._camera.calculateProjectionMatrix()
    }
  }

  /**
   * Render function
   *
   * @return {React.Fragment} React fragment
   */
  protected getDrawableComponents(): React.ReactElement[] {
    if (this._container !== null && this._viewerConfig !== null) {
      this._container.scrollTop = (this
        ._viewerConfig as ImageViewerConfigType).displayTop
      this._container.scrollLeft = (this
        ._viewerConfig as ImageViewerConfigType).displayLeft
    }

    const views: React.ReactElement[] = []
    if (this._viewerConfig !== null) {
      views.push(
        <HomographyCanvas
          key={`homographyCanvas${this.props.id}`}
          display={this._container}
          id={this.props.id}
        />
      )
      views.push(
        <Label3dCanvas
          key={`label3dCanvas${this.props.id}`}
          display={this._container}
          id={this.props.id}
          camera={this._camera}
        />
      )
    }

    return views
  }

  /** Menu components */
  protected getMenuComponents(): [] {
    return []
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
    const config = this._viewerConfig as HomographyViewerConfigType
    let zoomRatio = SCROLL_ZOOM_RATIO
    if (e.deltaY < 0) {
      zoomRatio = 1 / zoomRatio
    }
    const newDistance = config.distance * zoomRatio
    const newConfig = { ...config, distance: newDistance }
    Session.dispatch(changeViewerConfig(this._viewerId, newConfig))
  }
}

export default withStyles(viewerStyles)(HomographyViewer)

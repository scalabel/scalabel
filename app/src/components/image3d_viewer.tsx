import { withStyles } from "@material-ui/styles"
import * as React from "react"
import * as THREE from "three"

import Session from "../common/session"
import { IntrinsicCamera } from "../drawable/3d/intrinsic_camera"
import { isCurrentFrameLoaded } from "../functional/state_util"
import { viewerStyles } from "../styles/viewer"
import { Image3DViewerConfigType } from "../types/state"
import ImageCanvas from "./image_canvas"
import Label3dCanvas from "./label3d_canvas"
import { Viewer2D, Viewer2DProps } from "./viewer2d"

/**
 * Viewer for 3d labels on images
 */
class Image3DViewer extends Viewer2D {
  /** Intrinsic camera */
  private readonly _camera: IntrinsicCamera

  /**
   * Constructor
   *
   * @param {Object} props: react props
   * @param props
   */
  constructor(props: Viewer2DProps) {
    super(props)
    this._camera = new IntrinsicCamera()
    this._camera.up = new THREE.Vector3(0, -1, 0)
    this._camera.lookAt(new THREE.Vector3(0, 0, 1))
  }

  /** Component update function */
  public componentDidUpdate(): void {
    if (this._viewerConfig !== null) {
      const img3dConfig = this._viewerConfig as Image3DViewerConfigType
      const sensor = img3dConfig.sensor

      if (isCurrentFrameLoaded(this.state, img3dConfig.sensor)) {
        const image =
          Session.images[this.state.user.select.item][img3dConfig.sensor]
        this._camera.width = image.width
        this._camera.height = image.height
      }
      if (sensor in this.state.task.sensors) {
        this._camera.intrinsics = this.state.task.sensors[sensor].intrinsics
        const extrinsics = this.state.task.sensors[sensor].extrinsics
        this._camera.position.set(0, 0, 0)
        if (extrinsics !== null && extrinsics !== undefined) {
          const rotation4 = new THREE.Matrix4().makeRotationFromEuler(
            new THREE.Euler(
              -extrinsics.rotation.x,
              -extrinsics.rotation.y,
              -extrinsics.rotation.z
            )
          )
          const rotation = new THREE.Matrix3().setFromMatrix4(rotation4)
          const forward = new THREE.Vector3(0, 0, 1).applyMatrix3(rotation)
          const up = new THREE.Vector3(0, -1, 0).applyMatrix3(rotation)
          const newPos = new THREE.Vector3(
            extrinsics.translation.x,
            extrinsics.translation.y,
            extrinsics.translation.z
          )
          newPos.applyMatrix3(rotation).multiplyScalar(-1)
          this._camera.up.copy(up)
          this._camera.lookAt(forward)
          this._camera.position.copy(newPos)
        }
      }

      this._camera.calculateProjectionMatrix()

      if (Session.activeViewerId === this.props.id) {
        Session.label3dList.setActiveCamera(this._camera)
      }
    }
  }

  /**
   * Render function
   *
   * @return {React.Fragment} React fragment
   */
  protected getDrawableComponents(): React.ReactElement[] {
    const img3dConfig = this._viewerConfig as Image3DViewerConfigType
    if (this._container !== null && this._viewerConfig !== null) {
      this._container.scrollTop = img3dConfig.displayTop
      this._container.scrollLeft = img3dConfig.displayLeft
    }

    const views: React.ReactElement[] = []
    if (this._viewerConfig !== null) {
      views.push(
        <ImageCanvas
          key={`imageCanvas${this.props.id}`}
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
}

export default withStyles(viewerStyles)(Image3DViewer)

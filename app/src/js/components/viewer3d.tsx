import { withStyles } from '@material-ui/styles'
import React from 'react'
import * as THREE from 'three'
import { dragCamera, moveBack, moveCameraAndTarget, moveDown, moveForward, moveLeft, moveRight, moveUp, rotateCamera, updateLockStatus, zoomCamera } from '../action/point_cloud'
import Session from '../common/session'
import * as types from '../common/types'
import { PointCloudViewerConfigType } from '../functional/types'
import { Vector3D } from '../math/vector3d'
import { viewerStyles } from '../styles/viewer'
import { DrawableViewer, ViewerProps } from './drawable_viewer'
import Label3dCanvas from './label3d_canvas'
import PointCloudCanvas from './point_cloud_canvas'

/**
 * Viewer for images and 2d labels
 */
class Viewer3D extends DrawableViewer {
  /** Camera for math */
  private _camera: THREE.PerspectiveCamera
  /** Raycaster */
  private _raycaster: THREE.Raycaster

  /**
   * Constructor
   * @param {Object} props: react props
   */
  constructor (props: ViewerProps) {
    super(props)
    this._camera = new THREE.PerspectiveCamera(45, 1, 1, 1000)
    this._raycaster = new THREE.Raycaster()
  }

  /**
   * Render function
   * @return {React.Fragment} React fragment
   */
  protected getDrawableComponents () {
    const views: React.ReactElement[] = []
    if (this._viewerConfig) {
      views.push(
        <PointCloudCanvas
          key={`pointCloudCanvas${this.props.id}`}
          display={this._container}
          id={this.props.id}
        />
      )
      views.push(
        <Label3dCanvas
          key={`label3dCanvas${this.props.id}`}
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
    if (this._mouseDown) {
      if (this._mouseButton === 2) {
        this.updateCamera(this._viewerConfig as PointCloudViewerConfigType)
        Session.dispatch(dragCamera(
          oldX,
          oldY,
          this._mX,
          this._mY,
          this._camera,
          this._viewerId,
          this._viewerConfig as PointCloudViewerConfigType
        ))
      } else {
        Session.dispatch(rotateCamera(
          oldX,
          oldY,
          this._mX,
          this._mY,
          this._viewerId,
          this._viewerConfig as PointCloudViewerConfigType
        ))
      }
    }
  }

  /**
   * Handle double click
   * @param e
   */
  protected onDoubleClick (e: React.MouseEvent) {
    if (!this._container || !this._viewerConfig) {
      return
    }

    const normalized = this.normalizeCoordinates(e.clientX, e.clientY)
    this.updateCamera(this._viewerConfig as PointCloudViewerConfigType)

    const NDC = this.convertMouseToNDC(
      normalized[0],
      normalized[1]
    )

    this._raycaster.linePrecision = 0.2
    this._raycaster.setFromCamera(
      new THREE.Vector2(NDC[0], NDC[1]), this._camera
    )
    const pointCloud =
      Session.pointClouds[this._item][this._viewerConfig.sensor]

    const intersects = this._raycaster.intersectObject(pointCloud)

    if (intersects.length > 0) {
      const newTarget = intersects[0].point
      const viewerConfig = this._viewerConfig as PointCloudViewerConfigType
      Session.dispatch(moveCameraAndTarget(
        new Vector3D(
          viewerConfig.position.x - viewerConfig.target.x + newTarget.x,
          viewerConfig.position.y - viewerConfig.target.y + newTarget.y,
          viewerConfig.position.z - viewerConfig.target.z + newTarget.z
        ),
        new Vector3D(
          newTarget.x,
          newTarget.y,
          newTarget.z
        ),
        this._viewerId,
        viewerConfig
      ))
    }
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
      const pointCloudZoomAction = zoomCamera(
        e.deltaY,
        this._viewerId,
        this._viewerConfig as PointCloudViewerConfigType
      )
      if (pointCloudZoomAction) {
        Session.dispatch(pointCloudZoomAction)
      }
    }
  }

  /** Override key handler */
  protected onKeyDown (e: KeyboardEvent): void {
    const viewerConfig = this._viewerConfig as PointCloudViewerConfigType
    switch (e.key) {
      case types.Key.PERIOD:
        Session.dispatch(moveUp(this._viewerId, viewerConfig))
        break
      case types.Key.SLASH:
        Session.dispatch(moveDown(this._viewerId, viewerConfig))
        break
      case types.Key.S_LOW:
      case types.Key.S_UP:
        Session.dispatch(moveBack(this._viewerId, viewerConfig))
        break
      case types.Key.W_LOW:
      case types.Key.W_UP:
        Session.dispatch(moveForward(this._viewerId, viewerConfig))
        break
      case types.Key.A_LOW:
      case types.Key.A_UP:
        Session.dispatch(moveLeft(this._viewerId, viewerConfig))
        break
      case types.Key.D_LOW:
      case types.Key.D_UP:
        Session.dispatch(moveRight(this._viewerId, viewerConfig))
        break
      case types.Key.C_UP:
      case types.Key.C_LOW:
        Session.dispatch(updateLockStatus(this._viewerId, viewerConfig))
        break
    }
  }

  /** update camera parameters with config */
  private updateCamera (config: PointCloudViewerConfigType) {
    if (!this._viewerConfig) {
      return
    }

    if (this._container) {
      this._camera.aspect = this._container.offsetWidth /
        this._container.offsetHeight
      this._camera.updateProjectionMatrix()
    }

    const target = new THREE.Vector3(
      config.target.x,
      config.target.y,
      config.target.z
    )

    this._camera.up.x = config.verticalAxis.x
    this._camera.up.y = config.verticalAxis.y
    this._camera.up.z = config.verticalAxis.z
    this._camera.position.x = config.position.x
    this._camera.position.y = config.position.y
    this._camera.position.z = config.position.z
    this._camera.lookAt(target)
  }

  /**
   * Normalize coordinates to display
   * @param mX
   * @param mY
   */
  private convertMouseToNDC (mX: number, mY: number) {
    if (this._container) {
      let x = mX / this._container.offsetWidth
      let y = mY / this._container.offsetHeight
      x = 2 * x - 1
      y = -2 * y + 1

      return [x, y]
    }
    return [0, 0]
  }
}

export default withStyles(viewerStyles)(Viewer3D)

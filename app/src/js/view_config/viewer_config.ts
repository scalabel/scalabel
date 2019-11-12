import * as THREE from 'three'
import { changeViewerConfig } from '../action/common'
import { zoomImage } from '../action/image'
import { dragCamera, moveBack, moveCameraAndTarget, moveDown, moveForward, moveLeft, moveRight, moveUp, rotateCamera, updateLockStatus, zoomCamera } from '../action/point_cloud'

import Session from '../common/session'
import * as types from '../common/types'
import { ImageViewerConfigType, PointCloudViewerConfigType, State, ViewerConfigType } from '../functional/types'
import { Vector3D } from '../math/vector3d'
import { SCROLL_ZOOM_RATIO } from './image'

/**
 * Static class for updating viewer config in response to UI events
 */

/**
 * Normalize mouse coordinates to make canvas left top origin
 * @param x
 * @param y
 * @param canvas
 */
function normalizeCoordinatesToCanvas (
  x: number, y: number, container: HTMLDivElement): number[] {
  return [
    x - container.getBoundingClientRect().left,
    y - container.getBoundingClientRect().top
  ]
}

/**
 * Class for managing viewer config
 */
export default class ViewerConfigUpdater {
  /** The hashed list of keys currently down */
  private _keyDownMap: { [key: string]: boolean }
  /** Mouse x-coord */
  private _mX: number
  /** Mouse y-coord */
  private _mY: number
  /** Whether mouse is down */
  private _mouseDown: boolean
  /** which button is pressed on mouse down */
  private _mouseButton: number
  /** Display */
  private _container: HTMLDivElement | null
  /** Camera for math */
  private _camera: THREE.PerspectiveCamera
  /** Raycaster */
  private _raycaster: THREE.Raycaster
  /** target */
  private _target: THREE.Vector3
  /** viewer config */
  private _viewerConfig?: ViewerConfigType
  /** viewer id */
  private _viewerId: number
  /** item number */
  private _item: number

  constructor () {
    this._keyDownMap = {}
    this._mX = 0
    this._mY = 0
    this._container = null
    this._mouseDown = false
    this._camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000)
    this._raycaster = new THREE.Raycaster()
    this._target = new THREE.Vector3()
    this._mouseButton = 0
    this._item = -1
    this._viewerId = -1
  }

  /**
   * Set container
   * @param container
   */
  public setContainer (container: HTMLDivElement) {
    this._container = container
  }

  /**
   * Update state
   * @param state
   */
  public updateState (state: State, id: number) {
    this._viewerId = id
    this._viewerConfig = state.user.viewerConfigs[id]
    this._item = state.user.select.item
  }

  /**
   * Update state
   * @param state
   */
  public updateCamera (config: PointCloudViewerConfigType) {
    if (!this._viewerConfig) {
      return
    }

    if (this._container) {
      this._camera.aspect = this._container.offsetWidth /
        this._container.offsetHeight
      this._camera.updateProjectionMatrix()
    }
    this._target.x = config.target.x
    this._target.y = config.target.y
    this._target.z = config.target.z

    this._camera.up.x = config.verticalAxis.x
    this._camera.up.y = config.verticalAxis.y
    this._camera.up.z = config.verticalAxis.z
    this._camera.position.x = config.position.x
    this._camera.position.y = config.position.y
    this._camera.position.z = config.position.z
    this._camera.lookAt(this._target)
  }

  /**
   * Handle mouse movement
   * @param e
   */
  public onMouseMove (e: MouseEvent) {
    if (!this._container || !this._viewerConfig) {
      return
    }

    const normalized = normalizeCoordinatesToCanvas(
      e.clientX, e.clientY, this._container
    )
    if (this._mouseDown) {
      switch (this._viewerConfig.type) {
        case types.ViewerConfigType.IMAGE:
          if (this.isKeyDown(types.Key.META) ||
              this.isKeyDown(types.Key.CONTROL)) {
            const dx = normalized[0] - this._mX
            const dy = normalized[1] - this._mY
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
          break
        case types.ViewerConfigType.IMAGE_3D:
        case types.ViewerConfigType.POINT_CLOUD:
          if (this._mouseButton === 2) {
            this.updateCamera(this._viewerConfig as PointCloudViewerConfigType)
            Session.dispatch(dragCamera(
              this._mX,
              this._mY,
              normalized[0],
              normalized[1],
              this._camera,
              this._viewerId,
              this._viewerConfig as PointCloudViewerConfigType
            ))
          } else {
            Session.dispatch(rotateCamera(
              this._mX,
              this._mY,
              normalized[0],
              normalized[1],
              this._viewerId,
              this._viewerConfig as PointCloudViewerConfigType
            ))
          }
          break
      }
    }
    this._mX = normalized[0]
    this._mY = normalized[1]
  }

  /**
   * Handle mouse down
   * @param e
   */
  public onMouseDown (e: MouseEvent) {
    if (!this._container) {
      return
    }
    this._mouseDown = true
    this._mouseButton = e.button
    if (this._mouseButton === 2) {
      e.preventDefault()
    }
    const normalized = normalizeCoordinatesToCanvas(
      e.clientX, e.clientY, this._container
    )
    this._mX = normalized[0]
    this._mY = normalized[1]
  }

  /**
   * Handle mouse up
   * @param e
   */
  public onMouseUp (_e: MouseEvent) {
    this._mouseDown = false
  }

  /**
   * Handle double click
   * @param e
   */
  public onDoubleClick (e: MouseEvent) {
    if (!this._container || !this._viewerConfig) {
      return
    }

    const normalized = normalizeCoordinatesToCanvas(
      e.clientX, e.clientY, this._container
    )
    switch (this._viewerConfig.type) {
      case types.ViewerConfigType.POINT_CLOUD:
        this.updateCamera(this._viewerConfig as PointCloudViewerConfigType)

        const NDC = this.convertMouseToNDC(
          normalized[0],
          normalized[1]
        )
        const x = NDC[0]
        const y = NDC[1]

        this._raycaster.linePrecision = 0.2
        this._raycaster.setFromCamera(new THREE.Vector2(x, y), this._camera)
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
        break
    }
  }

  /**
   * Handle mouse wheel
   * @param _e
   */
  public onWheel (e: WheelEvent) {
    if (!this._viewerConfig) {
      return
    }

    switch (this._viewerConfig.type) {
      case types.ViewerConfigType.IMAGE:
        if (this.isKeyDown(types.Key.META) ||
            this.isKeyDown(types.Key.CONTROL)) {
          e.preventDefault()
          let zoomRatio = SCROLL_ZOOM_RATIO
          if (e.deltaY < 0) {
            zoomRatio = 1. / zoomRatio
          }
          const config = this._viewerConfig as ImageViewerConfigType
          const imageZoomAction = zoomImage(
            zoomRatio,
            this._viewerId,
            config
          )
          if (imageZoomAction) {
            Session.dispatch(imageZoomAction)
            if (this._container) {
              const displayLeft = zoomRatio * (this._mX + config.displayLeft) -
                this._mX
              const displayTop = zoomRatio * (this._mY + config.displayTop) -
                this._mY
              const newConfig = {
                ...this._viewerConfig,
                displayLeft,
                displayTop
              } as ImageViewerConfigType
              Session.dispatch(changeViewerConfig(
                this._viewerId,
                newConfig
              ))
            }
          }
        }
        break
      case types.ViewerConfigType.POINT_CLOUD:
        const pointCloudZoomAction = zoomCamera(
          e.deltaY,
          this._viewerId,
          this._viewerConfig as PointCloudViewerConfigType
        )
        if (pointCloudZoomAction) {
          Session.dispatch(pointCloudZoomAction)
        }
        break
    }
  }

  /**
   * Handle key down
   * @param e
   */
  public onKeyDown (e: KeyboardEvent) {
    const key = e.key
    this._keyDownMap[key] = true

    if (!this._viewerConfig) {
      return
    }

    switch (this._viewerConfig.type) {
      case types.ViewerConfigType.POINT_CLOUD:
        this.pointCloudKeyEvents(e.key)
        break
    }
  }

  /**
   * Handle key up
   * @param e
   */
  public onKeyUp (e: KeyboardEvent) {
    const key = e.key
    delete this._keyDownMap[key]
  }

  /**
   * Point cloud viewer config key events
   * @param key
   */
  private pointCloudKeyEvents (key: string) {
    const viewerConfig = this._viewerConfig as PointCloudViewerConfigType
    switch (key) {
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

  /**
   * Whether a specific key is pressed down
   * @param {string} key - the key to check
   * @return {boolean}
   */
  private isKeyDown (key: string): boolean {
    return this._keyDownMap[key]
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
